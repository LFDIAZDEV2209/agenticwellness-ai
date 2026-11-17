"""Contrato HTTP, esquema, fallback y cancelación; sin proveedor ni datos reales."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.core.config import get_settings
from app.schemas.nutrition_draft import NutritionDraftNarrative, NutritionDraftRequest
from app.services.nutrition_draft import generate_nutrition_draft, get_nutrition_model

URL = "/internal/nutrition/plan-draft"
PAYLOAD = {
    "targets": {"k": 1850, "c": 217, "p": 93, "f": 68, "fi": 26, "water_ml": 2500},
    "catalog_ids": ["avena"],
    "menus": [
        {"key": f"{m}:0", "name": "Menú", "items": [{"food_id": "avena", "grams": 40}]}
        for m in ("des", "alm", "mer", "cen")
    ],
}


@pytest.fixture
def client_and_model():
    model = Mock()
    model.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value={
            "narratives": {"es": "Propuesta", "en": "Proposal"},
            "instructions": {"es": "Revisar", "en": "Review"},
        }
    )
    app = create_app()
    app.dependency_overrides[get_nutrition_model] = lambda: model
    with TestClient(app) as client:
        yield client, model.with_structured_output.return_value.ainvoke


def test_internal_authentication(client_and_model):
    client, call = client_and_model
    assert client.post(URL, json=PAYLOAD).status_code == 401
    call.assert_not_called()


def test_bilingual_schema(client_and_model):
    client, _ = client_and_model
    response = client.post(
        URL, json=PAYLOAD, headers={"X-Internal-Key": get_settings().internal_api_key}
    )
    assert response.status_code == 200
    assert response.json()["source"] == "llm"
    assert set(response.json()["instructions"]) == {"es", "en"}
    # El LLM no reescribe los menús: se conserva la selección del motor C#.
    assert response.json()["menus"] == PAYLOAD["menus"]


def test_llm_only_returns_narrative_and_receives_compact_context(client_and_model):
    client, call = client_and_model
    client.post(URL, json=PAYLOAD, headers={"X-Internal-Key": get_settings().internal_api_key})
    schema = NutritionDraftNarrative.model_json_schema()
    assert "narratives" in schema["properties"] and "instructions" in schema["properties"]
    assert "menus" not in schema["properties"]
    prompt = call.await_args.args[0][1].content
    assert "avena" not in prompt and "food_id" not in prompt
    assert '"key": "des:0"' in prompt


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"narratives": {"es": "texto"}},
        {
            "narratives": {"es": "ES", "en": "EN"},
            "instructions": {"es": "ES", "en": "EN"},
            "menus": [
                {**m, "items": [{"food_id": "inventado", "grams": 40}]} for m in PAYLOAD["menus"]
            ],
        },
    ],
)
async def test_bad_output_falls_back_without_changing_deterministic_menus(result):
    model = Mock()
    model.with_structured_output.return_value.ainvoke = AsyncMock(return_value=result)
    response = await generate_nutrition_draft(NutritionDraftRequest.model_validate(PAYLOAD), model)
    assert response.source == "deterministic_fallback"
    assert [m.model_dump() for m in response.menus] == PAYLOAD["menus"]


@pytest.mark.parametrize("error", [RuntimeError("proveedor"), TimeoutError()])
async def test_provider_failure_uses_intended_fallback(error):
    model = Mock()
    model.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=error)
    response = await generate_nutrition_draft(NutritionDraftRequest.model_validate(PAYLOAD), model)
    assert response.source == "deterministic_fallback"


async def test_cancellation_propagates():
    model = Mock()
    model.with_structured_output.return_value.ainvoke = AsyncMock(
        side_effect=asyncio.CancelledError()
    )
    with pytest.raises(asyncio.CancelledError):
        await generate_nutrition_draft(NutritionDraftRequest.model_validate(PAYLOAD), model)


def test_unknown_food_rejected_before_llm(client_and_model):
    client, call = client_and_model
    response = client.post(
        URL,
        json={**PAYLOAD, "catalog_ids": ["otro"]},
        headers={"X-Internal-Key": get_settings().internal_api_key},
    )
    assert response.status_code == 422
    call.assert_not_called()
