"""Authenticated settings and explicitly requested synthetic connectivity check."""
from fastapi import APIRouter, Depends
from pydantic import Field, SecretStr
from app.api.routes import Payload, actor, services

router = APIRouter(prefix="/api/model-connection")


class ModelConnection(Payload):
    provider: str = "deepseek"
    base_url: str = Field(min_length=1, max_length=256)
    model: str = Field(min_length=1, max_length=128)
    api_key: SecretStr | None = None


@router.get("")
def get_connection(s=Depends(services), a=Depends(actor)):
    return s["model_connection"].get(a)


@router.put("")
def save_connection(body:ModelConnection,s=Depends(services),a=Depends(actor)):
    return s["model_connection"].save(a, **body.model_dump())


@router.post("/test")
def test_connection(s=Depends(services),a=Depends(actor)):
    return s["model_connection"].test(a)
