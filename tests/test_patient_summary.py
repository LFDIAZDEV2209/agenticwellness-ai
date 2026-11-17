"""Contrato camelCase, citas y fallback del resumen de paciente; sin proveedor real."""

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.core.config import get_settings
from app.services.patient_summary import get_patient_summary_model

URL = "/internal/program/patient-summary"
PAYLOAD = {
    "startDate": "2026-11-10",
    "endDate": "2026-11-16",
    "language": "es",
    "facts": [
        {
            "ref": "program.current",
            "section": "program",
            "text": "Estado registrado del programa: Active. Semana registrada: 1.",
        },
        {
            "ref": "nutrition.records",
            "section": "nutrition",
            "text": "Hay 3 registros de comida en 2 días del intervalo.",
        },
    ],
}
RESPONSE_KEYS = {
    "status",
    "kind",
    "isEstimate",
    "summary",
    "sourceRefs",
    "attention",
    "model",
    "generatedAt",
}


def _client(model):
    app = create_app()
    app.dependency_overrides[get_patient_summary_model] = lambda: model
    return TestClient(app)


def test_internal_authentication_required():
    with _client(None) as client:
        assert client.post(URL, json=PAYLOAD).status_code == 401


def test_without_model_returns_deterministic_summary_with_exact_camel_case():
    with _client(None) as client:
        response = client.post(
            URL,
            json=PAYLOAD,
            headers={"X-Internal-Key": get_settings().internal_api_key},
        )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == RESPONSE_KEYS
    assert body["status"] == "ok"
    assert body["kind"] == "patient_summary"
    assert body["isEstimate"] is True
    assert body["model"] is None
    assert body["sourceRefs"] == ["program.current", "nutrition.records"]
    assert "[program.current]" in body["summary"]
    assert "[nutrition.records]" in body["summary"]


def test_llm_response_is_returned_with_camel_case_aliases():
    model = Mock()
    model.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value={
            "summary": (
                "El programa está activo [program.current] y hay 3 registros [nutrition.records]."
            ),
            "source_refs": ["program.current", "nutrition.records"],
            "attention": [],
        }
    )
    with _client(model) as client:
        response = client.post(
            URL,
            json=PAYLOAD,
            headers={"X-Internal-Key": get_settings().internal_api_key},
        )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == RESPONSE_KEYS
    assert body["summary"].startswith("El programa está activo")
    assert body["sourceRefs"] == ["program.current", "nutrition.records"]


def test_invented_reference_falls_back_deterministically():
    model = Mock()
    model.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value={
            "summary": "Dato inventado [invented.ref].",
            "source_refs": ["invented.ref"],
            "attention": [],
        }
    )
    with _client(model) as client:
        response = client.post(
            URL,
            json=PAYLOAD,
            headers={"X-Internal-Key": get_settings().internal_api_key},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["sourceRefs"] == ["program.current", "nutrition.records"]
    assert "invented.ref" not in body["summary"]


@pytest.mark.parametrize(
    "payload",
    [
        {**PAYLOAD, "startDate": "2026-11-01"},
        {**PAYLOAD, "extra": "no permitido"},
        {**PAYLOAD, "facts": []},
    ],
)
def test_invalid_contract_is_rejected(payload):
    with _client(None) as client:
        response = client.post(
            URL,
            json=payload,
            headers={"X-Internal-Key": get_settings().internal_api_key},
        )
    assert response.status_code == 422
