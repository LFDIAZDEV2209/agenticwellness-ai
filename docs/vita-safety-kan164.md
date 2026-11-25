# KAN-164 · Validador de entrega de Vita

El chat de pacientes (JSON y SSE), WhatsApp y el adaptador de voz Vita aplican la
misma política local en español e inglés. ERP conserva su comportamiento.
No se añade un proveedor, un endpoint ni una dependencia del Motor 365.

## Entrega segura

- Los temas clínicos explícitos se rechazan sin invocar al proveedor de generación.
  Una lista de exclusión y un clasificador local por reglas revisan la respuesta
  completa. Los rechazos se reemplazan por un mensaje fijo que remite a un
  profesional. Una respuesta vacía o un error de generación recibe un mensaje
  fijo de reintento sin exponer el detalle del proveedor.
- No se emiten fragmentos del modelo. El SSE de texto conserva sus eventos de
  ciclo, pero publica el texto en un solo evento y lo repite como respuesta
  canónica en `done`. El adaptador de voz solo envía texto al TTS después de terminar
  el turno. Esto retrasa el primer texto/audio respecto al streaming progresivo.
- El modelo puede orientar hacia el plan publicado. No se le permite elegir
  objetivos de energía, indicar ejercicios concretos ni ajustar el plan. La lista
  cerrada de herramientas se aplica al ofrecerlas y al ejecutarlas. Una nueva
  herramienta del motor no hereda acceso de un perfil ERP ni del registro general.
- El audio directo de ElevenLabs en modo legado no pasa por este servicio y no
  queda validado. Con Vita activo, un cliente Vita no retrocede a ese modo
  si falta la ruta de voz segura. El corte histórico y el rollback completo al
  modo legado conservan sus contratos. KAN-162 y KAN-179 quedan fuera del cambio.

## Auditoría

`ai.vita_turns` conserva estado, motivo, canal y, cuando existe, la referencia
del mensaje visible de fallback. `ai.vita_turn_events` conserva una huella
`sha256:<hex>` del candidato bloqueado y su motivo. No se guardan el texto
rechazado, la excepción del proveedor ni el prompt en el log técnico. Tampoco
se guarda el candidato rechazado en el checkpoint del chat. La persistencia
usa el event log existente (`VITA_TURN_LOG_ENABLED`), sin migración adicional.
Una caída de la auditoría no habilita la entrega de un candidato rechazado.

Las reglas se pueden revisar en [vita_delivery.py](../app/safety/vita_delivery.py).
Son una defensa verificable, no una garantía frente a toda paráfrasis no
contemplada. Antes de ampliar el tráfico conviene probar mensajes en ambos idiomas
y medir la latencia y la tasa de falsos rechazos con voz real.

## Repetir las pruebas locales

- HTTP y checkpoint, sin proveedores externos:
  `ENVIRONMENT=test uv run python -m pytest -q tests/test_kan164_delivery.py tests/test_vita_voice.py`.
- Auditoría real: ejecutar `tests/test_vita_turn_log_postgres.py` con
  `VITA_TEST_DATABASE_URL` apuntando a una base local descartable cuyo nombre
  contenga `test`. Comprueba la fila y la huella en PostgreSQL sin texto clínico.
- Playwright: iniciar el proyecto paciente con Vite en `127.0.0.1:5191` y el
  fixture ASGI en otro proceso. El fixture no aplica migraciones ni usa la URL
  de base de datos. El navegador usa la interfaz real y envía su chat a la ruta
  SSE real del AI Service. Auth, Gateway y proveedor están simulados.

```sh
ENVIRONMENT=test AUTO_MIGRATE=false \
  DATABASE_URL=postgresql+psycopg://fixture:fixture@127.0.0.1:1/fixture \
  uv run python -m uvicorn tests.e2e.kan164_fixture:create --factory \
  --lifespan off --host 127.0.0.1 --port 8191
VITA_APP_DIR=/ruta/a/antares-paciente node tests/e2e/kan164_chat.mjs
VITA_APP_DIR=/ruta/a/antares-paciente VITA_BROWSER=webkit node tests/e2e/kan164_chat.mjs
```
