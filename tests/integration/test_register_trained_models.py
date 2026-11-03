"""Local registration preserves original data, versions and weight bytes."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest
from sqlalchemy import func, select

from app.adapters.model_features import CHANNEL_NAMES, FEATURE_PIPELINE_VERSION
from app.db import Store, observations
from app.scripts.register_trained_models import register_trained_models, FEATURE_POINTS


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False),encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def setup(tmp_path):
    runtime=tmp_path/"runtime"
    report=tmp_path/"report"
    runtime.mkdir()
    with sqlite3.connect(runtime/"app.db") as db:
        db.execute("create table preserved(value text)")
        db.execute("insert into preserved values('original')")
    store=Store(f"sqlite+pysqlite:///{(runtime/'app.db').as_posix()}")
    store.put("source","history-31002",dict(source_id="history-31002",face_id="31002",name="Original",timezone="+08:00"))
    store.put("point","31002:T0",dict(point_id="31002:T0",source_id="history-31002",face_id="31002",quantity="methane",unit="%"))
    with store.transaction() as connection:
        connection.execute(observations.insert().values(id="original-reading",point_id="31002:T0",source_id="history-31002",
            event_time="2022-01-01T00:00:00+00:00",available_at="2022-01-01T00:00:00+00:00",raw_value="0.1",value=.1,
            quality_flags=[],source_file_id="original-file",row_number=2,import_id="original-import"))
    store.close()
    january,february=tmp_path/"january.csv",tmp_path/"february.csv"
    header=','.join(["date",*CHANNEL_NAMES])+'\n'
    for path,month in [(january,1),(february,2)]:
        path.write_text(header+f"2022-{month:02}-05 10:00:00,0.2,1.2,-2,0.1,10,12\n2022-{month:02}-05 10:01:00,0.3,1.3,-3,0.2,11,13\n",encoding="utf-8")
    forecast,attribution=tmp_path/"forecast",tmp_path/"attribution"
    artifacts={}
    forecast.mkdir();attribution.mkdir()
    for name in ["checkpoint","scaler",*[f"residual_{i:02}" for i in range(1,13)]]:
        suffix=".pth" if name=="checkpoint" else ".json" if name=="scaler" else ".txt"
        file=forecast/(name+suffix);file.write_bytes(b"fixture model")
        artifacts[name]=dict(path=file.name,sha256=sha(file))
    write_json(forecast/"manifest.json",dict(kind="gas_forecast",model_id="fedformer-lgbm-31002",version="trained-v1",source_id="multisensor-31002-2022",face_id="31002",artifacts=artifacts,
        aggregation_seconds=300,input_window=90,horizon_seconds=3600,training_data=[dict(sha256=sha(january))],
        validation_assessment=dict(default_candidate="persistence",professional_mae_improved=False)))
    artifacts={}
    for name,suffix in [("classifier",".skops"),("training_metadata",".json")]:
        file=attribution/(name+suffix);file.write_bytes(b"fixture model")
        artifacts[name]=dict(path=file.name,sha256=sha(file))
    write_json(attribution/"manifest.json",dict(kind="gas_attribution",model_id="kmeans-svc-31002",version="trained-v1",artifacts=artifacts,automatic_attribution=True))
    catalog=tmp_path/"catalog.json"
    old=[]
    for index,kind in enumerate(["gas_forecast","gas_attribution"]):
        directory=tmp_path/f"old-{index}"
        write_json(directory/"manifest.json",dict(kind=kind,model_id=f"untrained-{index}",version="old-v1",artifacts={}))
        old.append(dict(bundle_dir=str(directory),source_id="history-31002",face_id="31002"))
    write_json(catalog,dict(models=old,python_executable="fixture-python.exe",timeout_seconds=120))
    return dict(january=january,february=february,forecast_bundle=forecast,attribution_bundle=attribution,
                runtime_dir=runtime,catalog=catalog,report_dir=report)


def count_source(runtime,source):
    store=Store(f"sqlite+pysqlite:///{(runtime/'app.db').as_posix()}")
    try:
        with store.transaction() as db:
            return db.execute(select(func.count()).select_from(observations).where(observations.c.source_id==source)).scalar_one()
    finally:
        store.close()


def test_registration_imports_six_channels_and_preserves_original_namespace(setup):
    original_hashes={str(p):sha(p) for folder in (setup['forecast_bundle'],setup['attribution_bundle']) for p in folder.iterdir() if p.name!='manifest.json'}
    result=register_trained_models(**setup)
    assert result['status']=='completed'
    assert count_source(setup['runtime_dir'],'multisensor-31002-2022')==24
    assert count_source(setup['runtime_dir'],'history-31002')==1
    with sqlite3.connect(setup['runtime_dir']/'app.db') as db:
        assert db.execute('select value from preserved').fetchone()==('original',)
    store=Store(f"sqlite+pysqlite:///{(setup['runtime_dir']/'app.db').as_posix()}")
    try:
        assert store.get('source','history-31002')['name']=='Original'
        assert store.get('point','31002:T0')['source_id']=='history-31002'
        points=[p for p in store.list('point') if p['source_id']=='multisensor-31002-2022']
        assert len(points)==6
        assert [p['point_id'] for p in points if p['quantity']=='methane']==['31002:multi:T0']
        assert all(p['lineage']=='processed' and p['unit_status']=='inferred' for p in points)
        assert store.get('model','fedformer-lgbm-31002:trained-v1')['default_eligible'] is False
        baseline=store.get('model','persistence-5min-max-31002:validated-20261103-v1')
        assert baseline['default_eligible'] is True and baseline['input_statistic']=='max'
    finally:
        store.close()
    assert {str(p):sha(p) for folder in (setup['forecast_bundle'],setup['attribution_bundle']) for p in folder.iterdir() if p.name!='manifest.json'}==original_hashes
    for folder in (setup['forecast_bundle'],setup['attribution_bundle']):
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        assert manifest['feature_points']==FEATURE_POINTS
        assert manifest['source_id']=='multisensor-31002-2022'
    assert json.loads((setup['attribution_bundle']/'manifest.json').read_text(encoding='utf-8'))['automatic_attribution'] is False


def test_repeat_is_idempotent_and_backup_is_not_overwritten(setup):
    first=register_trained_models(**setup)
    backup=Path(first['database_backup']['path'])
    backup_hash=sha(backup)
    second=register_trained_models(**setup)
    assert second['imported_rows_this_run']==0
    assert all(r['duplicate_file'] for r in second['imports'])
    assert sha(backup)==backup_hash
    assert second['database_backup']['created'] is False
    assert count_source(setup['runtime_dir'],'multisensor-31002-2022')==24
    assert len(json.loads(setup['catalog'].read_text(encoding='utf-8'))['models'])==4


def test_hash_mismatch_stops_before_database_or_catalog_mutation(setup):
    old_db=sha(setup['runtime_dir']/'app.db')
    old_catalog=setup['catalog'].read_bytes()
    (setup['forecast_bundle']/'checkpoint.pth').write_bytes(b'tampered')
    with pytest.raises(ValueError):
        register_trained_models(**setup)
    assert sha(setup['runtime_dir']/'app.db')==old_db
    assert setup['catalog'].read_bytes()==old_catalog


def test_different_existing_model_version_content_is_not_overwritten(setup):
    register_trained_models(**setup)
    store=Store(f"sqlite+pysqlite:///{(setup['runtime_dir']/'app.db').as_posix()}")
    existing=store.get('model','fedformer-lgbm-31002:trained-v1')
    existing['horizon_seconds']=300
    store.put('model','fedformer-lgbm-31002:trained-v1',existing)
    store.close()
    old_catalog=setup['catalog'].read_bytes()
    with pytest.raises(ValueError):
        register_trained_models(**setup)
    assert setup['catalog'].read_bytes()==old_catalog
    assert count_source(setup['runtime_dir'],'multisensor-31002-2022')==24


def test_manifest_binding_conflict_is_rejected_before_import(setup):
    path=setup['forecast_bundle']/'manifest.json'
    manifest=json.loads(path.read_text(encoding='utf-8'))
    manifest['feature_points']={**FEATURE_POINTS,CHANNEL_NAMES[0]:'another-source:T0'}
    write_json(path,manifest)
    with pytest.raises(ValueError):
        register_trained_models(**setup)
    assert count_source(setup['runtime_dir'],'multisensor-31002-2022')==0


def test_training_csv_digest_must_match_model_provenance(setup):
    setup['january'].write_text(setup['january'].read_text(encoding='utf-8').replace('0.2,','0.9,'),encoding='utf-8')
    with pytest.raises(ValueError):
        register_trained_models(**setup)
    assert count_source(setup['runtime_dir'],'multisensor-31002-2022')==0


def test_corrupt_existing_backup_is_not_silently_reused(setup):
    setup['report_dir'].mkdir()
    backup=setup['report_dir']/'before-trained-registration.sqlite'
    backup.write_bytes(b'corrupt backup')
    original_db=sha(setup['runtime_dir']/'app.db')
    with pytest.raises(ValueError):
        register_trained_models(**setup)
    assert backup.read_bytes()==b'corrupt backup'
    assert sha(setup['runtime_dir']/'app.db')==original_db
