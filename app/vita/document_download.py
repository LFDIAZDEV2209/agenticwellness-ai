"""Descarga y extracción privada de documentos del expediente."""

import json
import subprocess
import sys

import httpx
from anyio import to_thread

from app.core.config import Settings
from app.rag.extract import DocumentExtractionError, extract_text

MAX_DOCUMENT_BYTES = 5 * 1024 * 1024
MAX_EXCERPT_CHARS = 6_000
_LOCAL_STORAGE_PREFIX = "/api/v1/storage/"
_TEXT_EXTENSIONS = (".md", ".markdown", ".txt", ".text", ".rst")


def _download_url(raw: str, settings: Settings) -> httpx.URL:
    url = httpx.URL(raw)
    if url.userinfo or url.fragment:
        raise ValueError("document_origin_invalid")
    if url.is_relative_url and raw.startswith(_LOCAL_STORAGE_PREFIX):
        if (
            "\\" in url.path
            or {"..", "."}.intersection(url.path.split("/"))
            or not url.path.startswith(_LOCAL_STORAGE_PREFIX)
        ):
            raise ValueError("document_origin_invalid")
        if not url.params.get("sig") or not url.params.get("exp"):
            raise ValueError("document_signature_missing")
        return httpx.URL(settings.backend_base_url).join(url)

    allowed = httpx.URL(settings.vita_document_storage_origin)
    if (
        url.scheme == "https"
        and url.host
        and url.query
        and (url.scheme, url.host, url.port) == (allowed.scheme, allowed.host, allowed.port)
    ):
        return url
    raise ValueError("document_origin_invalid")


async def fetch_document_text(raw_url: str, settings: Settings) -> tuple[str, str, bool]:
    """Extrae con límites y cierra la descarga y el parser al cancelar el turno."""
    url = _download_url(raw_url, settings)
    # El parser y el cliente HTTP viven en un proceso efímero. Sus diagnósticos
    # pueden contener el contenido del documento o su URL. Solo vuelve el JSON.
    payload = json.dumps(
        {"url": str(url), "local_reference": raw_url, "timeout": settings.vita_http_timeout}
    ).encode()
    with subprocess.Popen(
        [sys.executable, "-m", "app.vita.document_download"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ) as worker:
        try:
            output, _ = await to_thread.run_sync(
                worker.communicate, payload, abandon_on_cancel=True
            )
        finally:
            if worker.poll() is None:
                worker.kill()
    if worker.returncode != 0:
        return "unavailable", "", False
    status, text, truncated = json.loads(output)
    return status, text, truncated


def _extract_document(url: str, local_reference: str, timeout: float) -> tuple[str, str, bool]:
    try:
        with (
            httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False) as client,
            client.stream("GET", url, headers={"Accept-Encoding": "identity"}) as response,
        ):
            if response.status_code != 200:
                return "unavailable", "", False
            content_type = response.headers.get("Content-Type", "").partition(";")[0].lower()
            data = bytearray()
            for chunk in response.iter_bytes():
                if len(data) + len(chunk) > MAX_DOCUMENT_BYTES:
                    return "too_large", "", False
                data.extend(chunk)
    except httpx.TimeoutException:
        return "timeout", "", False
    except (httpx.HTTPError, ValueError):
        return "unavailable", "", False

    path = httpx.URL(url).path.lower()
    if content_type == "application/pdf" or path.endswith(".pdf") or data.startswith(b"%PDF-"):
        filename = "document.pdf"
    elif content_type.startswith("text/") or path.endswith(_TEXT_EXTENSIONS):
        filename = "document.txt"
    else:
        return "no_text", "", False
    try:
        text = extract_text(filename, bytes(data)).strip()
    except DocumentExtractionError:
        return "no_text", "", False
    text = text.replace(url, "[enlace omitido]").replace(local_reference, "[enlace omitido]")
    truncated = len(text) > MAX_EXCERPT_CHARS
    return ("ok" if text else "no_text"), text[:MAX_EXCERPT_CHARS], truncated


if __name__ == "__main__":
    parameters = json.load(sys.stdin)
    print(json.dumps(_extract_document(**parameters)))
