"""Lecturas de ciclo, baterías y expediente con DTOs del contrato de Ola A."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from uuid import UUID

import httpx
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from app.core.config import get_settings
from app.graph.graph import build_graph
from app.observability.turn_trace import TurnTraceCollector, attach_collector
from app.vita import client, document_download, runtime
from app.vita.document_download import MAX_DOCUMENT_BYTES, MAX_EXCERPT_CHARS
from tests import test_vita_record_tools
from tests.fakes import RecordingFakeModel
from tests.test_extract import _synthetic_pdf
from tests.test_vita_record_tools import (
    AS_OF,
    KEY,
    OTHER,
    OWNER,
    graph_config,
    invoke_tool,
)

backend = test_vita_record_tools.backend

DOC_ID = "97609143-62df-4946-b3b9-01e11bd6f613"
SIGNED_PATH = "/api/v1/storage/clinical/record.txt?exp=1795000000&sig=fixture-capability"
DOCUMENT = {
    "id": DOC_ID,
    "name": "Informe de prueba",
    "type": "Informe",
    "date": AS_OF,
    "status": "Ready",
}
DOCUMENTS = {"items": [DOCUMENT], "total": 1, "page": 1, "pageSize": 20}
TICKET = {"document": DOCUMENT, "url": SIGNED_PATH, "expiresInSeconds": 120}
INTAKE = {
    "date": "2026-11-18",
    "mealCode": "alm",
    "calories": 420,
    "proteinG": None,
    "carbsG": None,
    "fatG": None,
    "fiberG": None,
    "waterMl": None,
    "source": "vita_chat",
}
NUTRITION = {
    "date": "2026-11-18",
    "plan": {
        "id": OTHER,
        "revision": 2,
        "startDate": "2026-11-01",
        "endDate": "2026-11-30",
        "days": [
            {
                "date": f"2026-11-{day:02}",
                "status": "planned" if day == 18 else "unavailable",
                "meals": [
                    {
                        "mealType": "lunch",
                        "description": "Menú de prueba",
                        "foods": "arroz",
                        "calories": 430,
                        "proteinG": 12.5,
                        "carbsG": None,
                        "fatG": None,
                        "fiberG": None,
                        "waterMl": None,
                        "notes": None,
                        "sortOrder": 1,
                        "suggestedTime": "12:30",
                        "notesEn": None,
                    }
                ]
                if day == 18
                else [],
            }
            for day in range(1, 31)
        ],
    },
    "recommendations": [{"cycleNumber": 1, "publishedAt": AS_OF, "patientMessage": "Nota visible"}],
    "history": [INTAKE],
}
ASSESSMENTS = {
    "batteries": [
        {
            "id": DOC_ID,
            "name": "Batería inicial",
            "status": "pending",
            "dueDate": AS_OF,
            "completedAt": None,
            "tests": [],
        },
        {
            "id": OTHER,
            "name": "Batería completada",
            "status": "completed",
            "dueDate": None,
            "completedAt": AS_OF,
            "tests": [
                {
                    "testName": "Test sintético",
                    "testCode": "fixture",
                    "completedAt": AS_OF,
                    "score": 7,
                    "scorePercentage": None,
                    "results": [
                        {
                            "resultType": "subscale",
                            "code": "x",
                            "label": "Subescala",
                            "value": 3.5,
                            "qualifier": None,
                            "severity": None,
                        },
                        {
                            "resultType": "total",
                            "code": "y",
                            "label": "Resultado",
                            "value": 7,
                            "qualifier": "Interpretación autorizada",
                            "severity": "mild",
                        },
                    ],
                },
            ],
        },
    ],
}


@pytest.fixture
def storage(monkeypatch):
    downloads = []
    response = {"status": 200, "body": b"Texto de prueba", "type": "text/plain"}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            downloads.append((self.path, self.headers.get("X-Internal-Key")))
            self.send_response(response["status"])
            self.send_header("Content-Type", response["type"])
            if "location" in response:
                self.send_header("Location", response["location"])
            content = response["body"]
            if callable(content):
                content = content(self.path)
            self.send_header("Content-Length", str(len(content)))
            with suppress(ConnectionError):
                self.end_headers()
                self.wfile.write(content)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    monkeypatch.setattr(
        get_settings(), "backend_base_url", f"http://127.0.0.1:{server.server_port}"
    )
    try:
        yield response, downloads
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize(
    "name,path,args,params,dto",
    [
        (
            "get_my_nutrition_plan",
            "nutrition",
            {"date": "2026-11-18"},
            {"date": "2026-11-18"},
            NUTRITION,
        ),
        ("get_my_assessment_results", "assessments", {}, {}, ASSESSMENTS),
        ("list_my_documents", "documents", {}, {"page": "1", "pageSize": "20"}, DOCUMENTS),
        (
            "list_my_documents",
            "documents",
            {"page": 9, "pageSize": 50},
            {"page": "9", "pageSize": "50"},
            {**DOCUMENTS, "page": 9, "pageSize": 50},
        ),
    ],
)
async def test_dto_read2_y_query(name, path, args, params, dto, backend):
    serve, calls = backend

    async def handler(_):
        return httpx.Response(200, json=dto)

    serve(handler)
    result = await invoke_tool(name, args)
    assert result == {"status": "ok", "data": dto}
    (request,) = calls
    assert request.url.path == f"/api/v1/internal/vita/{path}"
    assert dict(request.url.params) == {**params, "userId": OWNER}
    assert request.headers["X-Internal-Key"] == KEY


@pytest.mark.parametrize(
    "name,dto,text,status",
    [
        (
            "get_my_nutrition_plan",
            {"date": "2026-11-18", "plan": None, "recommendations": [], "history": []},
            "No hay un plan publicado",
            "empty",
        ),
        (
            "get_my_nutrition_plan",
            {"date": "2026-11-18", "plan": None, "recommendations": [], "history": [INTAKE]},
            "No hay un plan publicado",
            "ok",
        ),
        ("get_my_assessment_results", {"batteries": []}, "No hay baterías registradas.", "empty"),
        (
            "list_my_documents",
            {**DOCUMENTS, "items": [], "total": 0},
            "No hay documentos registrados.",
            "empty",
        ),
        (
            "list_my_documents",
            {**DOCUMENTS, "items": [], "page": 2},
            "No hay documentos en esta página.",
            "empty",
        ),
        (
            "get_my_medications",
            {
                "asOf": AS_OF,
                "localDate": "2026-11-18",
                "timeZone": "America/Bogota",
                "treatments": [],
            },
            "No hay tratamientos vigentes.",
            "empty",
        ),
    ],
)
async def test_ausencia_explicita_sin_inventar_datos(name, dto, text, status, backend):
    serve, _ = backend

    async def handler(_):
        return httpx.Response(200, json=dto)

    serve(handler)
    result = await invoke_tool(name)
    assert result["status"] == status and text in result["message"]
    assert result["data"] == dto


async def test_bateria_pendiente_y_subescala_conservan_solo_lo_autorizado(backend):
    serve, _ = backend
    pending, completed = ASSESSMENTS["batteries"]
    (test,) = completed["tests"]
    subscale, total = test["results"]
    contaminated = {
        "batteries": [
            {**pending, "tests": [{"answer": "texto privado"}]},
            {
                **completed,
                "tests": [
                    {
                        **test,
                        "answers": ["texto privado"],
                        "results": [
                            {**subscale, "qualifier": "texto privado", "severity": "texto privado"},
                            total,
                        ],
                    }
                ],
            },
        ]
    }

    async def handler(_):
        return httpx.Response(200, json=contaminated)

    serve(handler)
    assert (await invoke_tool("get_my_assessment_results"))["data"] == ASSESSMENTS


@pytest.mark.parametrize(
    "name,args,expected",
    [
        ("get_my_nutrition_plan", {"date": "2026-02-30"}, "invalid_nutrition_date"),
        ("list_my_documents", {"page": 0}, "invalid_document_arguments"),
        ("list_my_documents", {"pageSize": 51}, "invalid_document_arguments"),
        ("read_my_document", {"id": "../foreign"}, "invalid_document_arguments"),
        ("read_my_document", {"id": DOC_ID, "patient_id": OTHER}, "tool_not_authorized"),
    ],
)
async def test_argumentos_invalidos_no_emiten_http(name, args, expected, backend):
    _, calls = backend
    assert (await invoke_tool(name, args))["code"] == expected
    assert calls == []


@pytest.mark.parametrize(
    "status,payload,code",
    [
        (403, {"detail": "VITA_PATIENT_UNAVAILABLE"}, "vita_patient_unavailable"),
        (404, None, "vita_module_unavailable"),
        (503, {"code": "CONSENT_DOCUMENT_UNAVAILABLE"}, "consent_document_unavailable"),
        (400, None, "invalid_nutrition_date"),
    ],
)
async def test_errores_read2_son_resultados_seguros(status, payload, code, backend):
    serve, _ = backend

    async def handler(_):
        return httpx.Response(status, json=payload)

    serve(handler)
    result = await invoke_tool("get_my_nutrition_plan", {"date": "2026-11-18"})
    assert result["code"] == code and "data" not in result


async def test_documento_descarga_real_trunca_y_redacta_capacidad(backend, storage, caplog):
    serve, calls = backend
    response, downloads = storage
    response["body"] = (SIGNED_PATH + "\n" + "contenido clínico sintético " * 500).encode()

    async def handler(_):
        return httpx.Response(200, json=TICKET)

    serve(handler)
    caplog.set_level(logging.DEBUG)
    collector = TurnTraceCollector(thread_id="fixture", channel="text")
    config = attach_collector(graph_config(), collector)
    result = await invoke_tool("read_my_document", {"id": DOC_ID}, config=config)
    assert result["status"] == "ok"
    assert result["data"]["document"] == DOCUMENT
    assert result["data"]["truncated"] is True
    assert len(result["data"]["text"]) == MAX_EXCERPT_CHARS
    assert downloads == [(SIGNED_PATH, None)]
    assert calls[0].url.params["userId"] == OWNER and calls[0].headers["X-Internal-Key"] == KEY
    assert "contenido clínico sintético" in result["data"]["text"]
    trace = json.dumps(collector.finish(status="completed").to_log())
    assert "fixture-capability" not in json.dumps(result) + trace + caplog.text
    assert "contenido clínico sintético" not in trace + caplog.text


async def test_documento_pdf_con_texto_extraible(backend, storage):
    serve, _ = backend
    response, _ = storage
    response.update({"type": "application/pdf", "body": _synthetic_pdf(encrypted=True)})

    async def handler(_):
        return httpx.Response(200, json=TICKET)

    serve(handler)
    result = await invoke_tool("read_my_document", {"id": DOC_ID})
    assert result["data"] == {"document": DOCUMENT, "text": "antes despues", "truncated": False}


@pytest.mark.parametrize(
    "case,expected",
    [
        ("broken_pdf", "vita_document_no_text"),
        ("empty", "vita_document_no_text"),
        ("too_large", "vita_document_too_large"),
        ("redirect", "vita_document_unavailable"),
        ("foreign", "vita_document_unavailable"),
        ("unauthorized_origin", "vita_document_unavailable"),
        ("malformed_url", "vita_document_unavailable"),
        ("wrong_document", "vita_document_unavailable"),
    ],
)
async def test_documento_sin_texto_o_enlace_invalido_se_degrada(case, expected, backend, storage):
    serve, _ = backend
    response, downloads = storage
    ticket = TICKET.copy()
    if case == "broken_pdf":
        response.update({"type": "application/pdf", "body": b"%PDF-1.7 fixture-capability"})
    if case == "empty":
        response["body"] = b" "
    if case == "too_large":
        response["body"] = b"x" * (MAX_DOCUMENT_BYTES + 1)
    if case == "redirect":
        response.update({"status": 302, "location": "/never"})
    if case == "unauthorized_origin":
        ticket["url"] = "https://storage.invalid/object?sig=other"
    if case == "malformed_url":
        ticket["url"] = "https://[broken/"
    if case == "wrong_document":
        ticket["document"] = {**DOCUMENT, "id": OTHER}

    async def handler(_):
        return httpx.Response(404) if case == "foreign" else httpx.Response(200, json=ticket)

    serve(handler)
    result = await invoke_tool("read_my_document", {"id": DOC_ID})
    assert result["code"] == expected and "data" not in result
    if case in ("foreign", "unauthorized_origin", "malformed_url", "wrong_document"):
        assert downloads == []
    elif case == "redirect":
        assert downloads == [(SIGNED_PATH, None)]


async def test_voz_puede_listar_y_leer_en_el_mismo_turno(backend, storage, monkeypatch):
    serve, calls = backend
    _, downloads = storage

    async def handler(request):
        path = request.url.path
        payload = (
            [] if path.endswith("/agents") else DOCUMENTS if path.endswith("/documents") else TICKET
        )
        return httpx.Response(200, json=payload)

    serve(handler)
    model = RecordingFakeModel(
        responses=[
            AIMessage(
                content="", tool_calls=[{"name": "list_my_documents", "args": {}, "id": "list"}]
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "read_my_document", "args": {"id": DOC_ID}, "id": "read"}],
            ),
            AIMessage(content="Respuesta oral de prueba."),
        ]
    )
    monkeypatch.setattr(runtime, "get_chat_model", lambda **_: model)

    @asynccontextmanager
    async def unlocked(_):
        yield

    async def no_kbs(*_):
        return []

    monkeypatch.setattr("app.vita.locking.thread_guard", unlocked)
    monkeypatch.setattr(runtime, "resolve_knowledge_base_ids", no_kbs)
    config = graph_config("voice")
    message = "Lee mi documento más reciente"
    turn = await runtime.resolve_turn(
        user_id=OWNER,
        text=message,
        session=None,
        config=config,
        channel="voice",
        base_graph=build_graph(model=model, checkpointer=MemorySaver()),
    )
    updates = [event async for event in turn.graph.astream({"input": message}, config=config)]
    assert [r.url.path.rsplit("/", 1)[-1] for r in calls] == ["agents", "documents", DOC_ID]
    assert downloads == [(SIGNED_PATH, None)]
    assert updates[-1]["answer"]["messages"][-1].content == "Respuesta oral de prueba."


async def test_timeout_y_cancelacion_del_documento(backend, monkeypatch):
    serve, _ = backend
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def handler(_):
        return httpx.Response(200, json=TICKET)

    async def fetch(*_):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    serve(handler)
    monkeypatch.setattr(client, "fetch_document_text", fetch)
    settings = get_settings()
    monkeypatch.setattr(settings, "vita_http_timeout", 0.05)
    assert (await invoke_tool("read_my_document", {"id": DOC_ID}))["code"] == "tool_timeout"
    assert cancelled.is_set()
    started.clear()
    cancelled.clear()
    monkeypatch.setattr(settings, "vita_http_timeout", 3.0)
    task = asyncio.create_task(client.read_document(OWNER, UUID(DOC_ID)))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


async def test_cancelar_descarga_real_cierra_el_proceso(backend, storage, monkeypatch):
    serve, _ = backend
    response, downloads = storage
    release = Event()

    def wait_to_send(_):
        release.wait(timeout=5)
        return b"fixture text"

    response["body"] = wait_to_send

    async def handler(_):
        return httpx.Response(200, json=TICKET)

    serve(handler)
    workers = []
    launch = document_download.subprocess.Popen

    def launch_tracked(*args, **kwargs):
        worker = launch(*args, **kwargs)
        workers.append(worker)
        return worker

    monkeypatch.setattr(document_download.subprocess, "Popen", launch_tracked)
    task = asyncio.create_task(client.read_document(OWNER, UUID(DOC_ID)))
    try:
        async with asyncio.timeout(2):
            while not downloads:
                await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(workers) == 1 and workers[0].poll() is not None
    finally:
        release.set()
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
