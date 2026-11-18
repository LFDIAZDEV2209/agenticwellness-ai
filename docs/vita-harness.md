# Auditoría del harness Vita · Ola A

El harness ejecuta guardrails, prepara el contexto, aplica el registro de herramientas y registra
una traza por turno. El grafo reactivo ofrece herramientas al modelo. La variante de respuesta
directa decide las lecturas en código y reserva el modelo para la respuesta. Ambas reutilizan
los nodos, el checkpointer y las políticas de ejecución. [Ensamblado del grafo](../app/graph/graph.py)

## Matriz por canal

| Canal y entrada | Perfil y herramientas inyectadas | Límites en AI Service | Salida y modos de fallo |
| --- | --- | --- | --- |
| App · `POST /api/v1/chat/stream` con `X-Internal-Key`. `supports_vita` y el flag del runtime activan el catálogo de paciente. | Vita resuelve un perfil publicado y autorizado o el asistente general. Inyecta `get_my_day`, `get_my_nutrition`, `get_my_assessments`, `get_my_podcast`, `get_my_exercise` y `get_my_progress`, o la lectura del ámbito detectado. RAG usa las bases de conocimiento autorizadas. El camino legado usa el perfil solicitado o el supervisor. `mark_control_declined` se usa en la app con un contexto de control activo. | Generación de Vita 20 s por defecto. Camino legado 90 s. Lecturas HTTP de Vita 3 s por defecto y clasificador opcional 2 s. | SSE `start`, `message`, `node`, `flow`, `done` y, cuando corresponde, `control_signal`. `done` transporta acciones, fuentes y referencias de ejecución. Sin identidad en el flujo Vita se rechaza con 403. Un hilo ocupado en la preparación puede devolver 409. Fallos de runtime previos al stream pueden devolver 503. Un fallo en generación emite `error`. |
| Voz · `POST /internal/voice/v1/chat/completions` con bearer de servicio, sesión firmada, conversación y contador. | Selección con el mismo catálogo de paciente que el chat Vita. Ejecuta lecturas decididas en código antes del nodo `answer` y RAG autorizado si aplica. El modelo final recibe sus resultados sin tools enlazadas. El cambio de idioma es un control del adaptador. | Hasta 20 s por defecto para esperar una admisión o ejecutar la generación, por separado. El proveedor tiene su propio límite de cascada. | SSE compatible con OpenAI, con `chat.completion.chunk` y `[DONE]`. Reintentos del mismo turno pueden reutilizar su respuesta. Firma, sesión o vínculo inválidos devuelven 401/403 según el control. Un conflicto de admisión o hilo puede devolver 409. Fallos de preparación o espera pueden devolver 503. Una interrupción durante generación emite un código fijo en SSE. |
| WhatsApp · el backend llama a `POST /api/v1/chat` con `agent=whatsapp`. | Perfil legado `whatsapp` con prompt breve de educación en salud y dos tools, `calculate` y `get_current_time`. El overlay de lecturas de app y voz queda pendiente para este perfil. | Generación legada 90 s por defecto en el AI Service. El transporte del backend tiene su propio plazo. | JSON de `ChatResponse` con respuesta, `thread_id`, herramientas y `execution_id`. El guard de catálogo devuelve texto seguro. Fallos de proveedor o timeout conservan el error HTTP del adaptador. La entrega de la respuesta de respaldo pertenece al backend de WhatsApp. |

Rutas y configuración de la matriz. [Chat SSE y JSON](../app/api/routes/chat.py) ·
[Custom LLM de voz](../app/api/routes/vita_voice.py) · [Resolución Vita](../app/vita/runtime.py) ·
[Overlay por perfil](../app/agents/runtime_registry.py) · [Perfiles legados](../app/agents/registry.py) ·
[Configuración](../app/core/config.py)

En la prueba con audio real de Ola 0 hubo callbacks concurrentes durante una admisión activa,
seguidos de 409 y cierre del Custom LLM. Se desactivaron los turnos especulativos en la versión
dedicada del proveedor y se amplió su límite de cascada a 8 s. También se observó `vita_thread_busy`
al sincronizar un hilo ya existente. La prueba local reprodujo un lock retenido al desconectar el
stream. Los 409 históricos pueden tener ambas causas. La corrección de admisión y reanudación
corresponde a 2.2. [Evidencia de voz y locks](vita-runtime.md)

## Presupuesto y continuidad

- `VITA_CONTEXT_TOKEN_BUDGET` limita el prompt a 12.000 tokens estimados por defecto. El cálculo
  incluye políticas de sistema, identidad, memoria, snapshot de lecturas preparadas, historial y
  resultados de herramientas. Presenta los turnos recientes íntegros y resume por extracción los
  más antiguos. Conserva la entrada vigente y deja juntas cada llamada y su resultado. Un lote
  irreducible puede salir del prompt completo. El checkpoint conserva el historial original.
  La estimación es independiente del proveedor y excluye los esquemas de tools enviados por separado.
  [Context builder](../app/vita/context_builder.py) · [Nodos del grafo](../app/graph/nodes.py)
- El catálogo del perfil y el registro tipado se aplican tanto al enlace al modelo como a la
  ejecución. Se controla el máximo por herramienta y por turno. Los reintentos por timeout se
  reservan a herramientas idempotentes y comparten el deadline de la llamada. Al agotar el presupuesto
  global, el grafo cierra con una respuesta útil. Las tres lecturas nuevas quedan reservadas en
  el registro. [Política de tools](../app/tools/policy.py)

## Event log

`ai.vita_turns` contiene la referencia al mensaje, canal, estado, tiempos, tokens y costo estimado.
`ai.vita_turn_events` registra nodos, llamadas a tools, uso del modelo y decisiones de política en
orden de finalización. El turno y sus eventos se insertan en una transacción. La BD rechaza `UPDATE`
y permite eliminación para retención. La traza técnica excluye texto clínico, argumentos,
resultados, excepciones libres, firmas e identificadores de paciente e hilo en los logs. Los
identificadores necesarios para enlazar el turno permanecen en PostgreSQL.
[Migración](../alembic/versions/5a7c3e9d1b42_vita_turn_event_log.py) · [Recolector](../app/observability/turn_trace.py)

La persistencia es best effort con un plazo de escritura de un segundo. `VITA_TURN_LOG_ENABLED`
controla la escritura en PostgreSQL. El log técnico conserva el resumen si falla la BD. La traza
empieza con el turno de grafo en chat y WhatsApp y con la
admisión de generación en voz. Los replays reutilizan el turno de voz. `downgrade -1` elimina el
event log y sus datos. La aceptación de audio remoto y de entrega entrante por WhatsApp requiere
probar con los proveedores.
[Pruebas del harness](../tests/test_vita_harness.py) · [Prueba de PostgreSQL y rollback](../tests/test_vita_turn_log_postgres.py)
