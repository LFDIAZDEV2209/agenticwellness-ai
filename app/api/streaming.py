"""Cierra streams suspendidos también si ASGI cancela el envío al cliente."""

from anyio import CancelScope
from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send


class ClosingStreamingResponse(StreamingResponse):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # La desconexión puede ocurrir en send(), fuera del generador. ASGI
            # no cierra automáticamente un body_iterator suspendido en yield.
            close = getattr(self.body_iterator, "aclose", None)
            if close is not None:
                with CancelScope(shield=True):
                    await close()
