# Admisión durable de utterances nativas Vita

## Evidencia y alcance

`system__agent_turns` cuenta turnos **del agente**; no es un ordinal de utterances
del paciente ni una clave idempotente de callbacks. Las herramientas nativas
modifican el estado conversacional. Fuentes primarias:

- https://elevenlabs.io/docs/eleven-agents/customization/personalization/dynamic-variables
- https://elevenlabs.io/docs/eleven-agents/customization/llm/custom-llm

La auditoría QA comunicada por root encontró dos segmentos reales del audio
sintético (48 y 22 caracteres), ambos iguales a las dos frases enviadas. El segundo
no exporta `source_medium`; eso **no** lo convierte en control. El snapshot final
contiene dos usuarios y `system__agent_turns=1`. No acredita por sí solo los
contadores de cada callback ni una reescritura del prefijo. No se descartan mensajes
por ausencia de medium, por su longitud ni por contener una instrucción breve.

Esta implementación admite contadores repetidos como metadatos y conserva el
prefijo canónico exacto. No intenta reparar un prefijo alterado ni inventa contadores.
Root conserva su `composed_input` para routing/prefetch y el Human visible original
del último segmento; este carril no modifica route, language, runtime ni nodos.

## API interna de admisión

La verificación HMAC, audiencia/expiración, sesión/conversación, consentimiento y
rate limit del adaptador siguen siendo obligatorias y anteriores a esta API.
`VoiceBinding` se construye exclusivamente del registro ya verificado:

```python
from app.voice.store import VoiceBinding, voice_store
from app.voice.turns import AdmittedVoiceGraph, prepare_admitted_turn

binding = VoiceBinding(record.id, record.user_id, record.thread_id, record.conversation_id)
ordinal, prefix, previous = history_identity(canonical_messages)
admission = await voice_store.admit_turn(
    binding, ordinal, x_vita_turn, prefix, previous,
    phase="control" if actual_language_switch else "utterance",
)
```

El body no proporciona identidad ni `phase`. Root decide `actual_language_switch`
con su helper validado; el idioma inicial fiable de APP evita llamadas redundantes.
No llamar a `start_turn` en el camino nuevo: se conserva para consumidores antiguos
y sus garantías anteriores, hasta integrar los hooks. `provider_turn` admite cero y
no se usa en decisiones de generación/replay; root debe aceptar cero también en
el Header si el proveedor lo envía. Se valida ordinal positivo y hashes hex de 64
caracteres. El primer `previous` debe ser SHA256 de `[]`, igual a `history_identity`.

La fila de sesión `FOR UPDATE` serializa admisión, revocación y fencing. La PK
`(session_id, ordinal)` y la comparación de prefijo exacto siguen siendo autoridad.
En duplicados se valida también el prefijo de la fila anterior. Acciones:

| action | Significado y responsabilidad de root |
| --- | --- |
| `run` | Único propietario de generación de esa utterance. Solo esta acción construye el grafo ejecutable. |
| `wait` | Duplicado de un `running`. Esperar fuera de transacción con `wait_for_turn`; jamás ejecutar otro LLM. |
| `replay` | Recibo completado todavía vigente. Leer su `response_message_id` del checkpoint bajo guarda. |
| `obsolete` | Hay un ordinal posterior o el recibo quedó interrumpido/supersedido. Terminar sin texto ni otra generación. |
| `control` | Reserva de la misma utterance para un cambio real de idioma. Solo emitir la llamada nativa; sin LLM ni mensaje Human adicional. |

Un `control` nuevo reserva la fila con estado `control`. Su retorno con la misma
utterance/prefijo convierte esa fila a `running` una única vez y devuelve `run`.
Un duplicado concurrente recibe `wait`. Un control tardío que ya pasó a `running`
es `obsolete`, no otra llamada nativa. El ID de respuesta permanece estable.

Una utterance nueva con ordinal siguiente y prefijo válido cambia el `running` o
`control` anterior a `superseded` y reserva el nuevo. Una validación fallida no
supersede nada. Recibos `completed` anteriores tampoco se reproducen cuando ya hay
un ordinal posterior. Un recibo interrumpido nunca se reclama de nuevo para generar;
la utterance siguiente válida sí puede continuar. Frases repetidas en ordinales
distintos siguen siendo distintas, sin deduplicación por igualdad de texto.

`VoiceContextError.metadata` y `TurnAdmission.metadata` solo contienen
`previous_hash_matches` y `counter_reused`, booleanos calculados en admisión. Root
puede registrar esos flags junto al código; nunca registrar la admisión completa,
binding, hashes, firma, contenido o IDs del paciente. Identidad inválida no expone
diagnóstico sobre filas ajenas. Errores de prefijo/secuencia conservan 409.

## Hooks exactos en `vita_voice.py` (solo root)

1. Sustituir la reserva antigua por `admit_turn` **antes** de preparar historial o
   tomar el lock de ejecución. Así la nueva utterance puede superseder al viejo
   stream, que aún mantiene ese lock. No llamar después a `start_turn`.

2. Si `wait`, usar `admission = await voice_store.wait_for_turn(admission,
   timeout=settings.vita_turn_timeout)`. El resultado es `replay` u `obsolete`,
   nunca `run`. Espera cancelable, con sondeo de 50 ms; ninguna conexión/transacción
   se retiene entre sondeos. Revocación, borrado y expiración abortan. Un timeout
   no autoriza regenerar ni reclamar el recibo.

3. Para `obsolete` devolver un final SSE sin contenido (`[DONE]`, con el cierre
   OpenAI que use el adaptador). No convertir una supersesión esperada en una
   nueva generación o un error fatal del proveedor. No usar la respuesta vieja.

4. Construir `config` como ahora, incluyendo `user_id`, `thread_id` interno,
   `voice_session_id`, IDs de mensajes y canal. Para `run/control/replay`, llamar
   **antes de routing**:

   ```python
   await prepare_admitted_turn(voice_store, admission, config, base_graph.checkpointer,
                               timeout=settings.vita_turn_timeout)
   ```

   Esto reemplaza `history_service.prepare_turn` en el camino nuevo. Espera el mismo
   advisory lock hasta que el stream supersedido cierre; nunca retiene la fila SQL
   mientras espera. Comprueba retención mensual/sesión y captura revisión. Un
   DELETE entre preparación y ejecución invalida el request, sin restaurar contexto.

5. Para `control`, mantener el generador de llamada de idioma bajo `thread_guard` y
   revalidar `voice_store.check_current(admission)` e historial antes de emitirla.
   No llamar `finish_admission` ni generar respuesta en este camino. La siguiente
   petición de la misma utterance con resultado de la tool usa `phase="utterance"`.

6. Para `replay`, mantener la guarda de historial existente bajo `thread_guard`,
   añadir `await voice_store.check_current(admission, completed_ok=True)` y releer
   checkpoint dentro de ese lock. Usar `admission.response_message_id`. Un waiter
   resuelto nunca pasa por `resolve_turn` ni por el LLM.

7. Para `run`, resolver el runtime una sola vez y reemplazar `turn.graph` en la
   ejecución por este wrapper (sin mutar el grafo cacheado ni añadir otro lock):

   ```python
   graph = AdmittedVoiceGraph(turn.graph, voice_store, admission,
                             wait_timeout=settings.vita_turn_timeout)
   stream = graph.astream(inputs, config=config, stream_mode=["messages", "values"])
   ```

   El wrapper quita únicamente el `LockedGraph` exterior y vuelve a tomar el mismo
   lock, con espera cancelable. Clona superficialmente el grafo para instalar un
   saver por admisión. La revisión/sesión se comprueban antes de ejecutar.
   Continúa filtrando tokens del nodo `answer` exactamente como ahora.

8. El wrapper sondea vigencia también durante un LLM/tool que no emite datos,
   cancela la tarea obsoleta, espera su cierre y libera el lock antes de ejecutar
   el nuevo turno. Ante `voice_turn_obsolete`, tratar el stream viejo como cierre
   esperado sin contenido nuevo. Mantener las comprobaciones actuales de catálogo,
   sesión, desconexión y el `CancelScope(shield=True)` de limpieza del adaptador.

9. Sustituir finalización antigua por
   `await voice_store.finish_admission(admission, "completed")`. Solo si devuelve
   `True`, marcar completado y emitir cierre de éxito. Si devuelve `False`, otro
   turno ya ganó: no emitir éxito viejo. En el `finally` del **propietario run**,
   finalizar como `interrupted` si no completó. Nunca finalizar desde wait/replay/
   control. La limpieza de un runner viejo no puede modificar al sucesor.

El wrapper protege `aput` y `aput_writes`, incluidos namespaces de subgrafos, con
una guarda SQL por escritura corta. Esa guarda y la admisión toman la misma fila
de sesión: una escritura previa termina antes del commit de supersesión; después
de ese commit no se acepta otra escritura vieja. No basta con sondear únicamente
antes de enviar tokens, porque el checkpoint puede escribir en segundo plano.
Conservar el saver canónico de `base_graph` para historial/borrado/importación;
no sustituirlo globalmente por el saver de una admisión.

## Migración, despliegue y rollback

`94e1c2b75a06`, padre `18c4f6a91b20`, elimina únicamente
`uq_vita_voice_turns_session_id` (UNIQUE de session/provider_turn). Conserva todas
las columnas, la PK ordinal, el FK cascade y los recibos. No hay backfill ni seeds.
Aplicar antes de activar los hooks. Evitar mezclar runners nuevos con binarios
viejos sin fencing: activar la ruta nueva cuando todas las réplicas estén listas.

`downgrade 18c4f6a91b20` toma un lock de tabla y comprueba duplicados antes de
recrear UNIQUE. Si hay repeticiones, falla con `voice_counter_duplicates`, conserva
filas y versión y no borra/deduplica nada. Se puede mantener este esquema y volver
al flag de voz legacy sin hacer downgrade. Para un downgrade posterior, esperar el
ciclo autorizado de retención/expiración y coordinar mantenimiento; no borrar
recibos para hacer pasar una migración.

Un runner caído conserva un recibo `running`: los duplicados esperan hasta timeout
sin volver a generar. No se roba el claim con un lease, porque no se puede probar
que el LLM anterior terminó. La utterance siguiente válida puede supersederlo. La
cancelación inmediata presupone que el runner usa el wrapper; un código anterior
sin fencing no adquiere estas garantías por cambiar solo la migración.

## Verificación

`tests/test_voice_native_turns_postgres.py` reutiliza la fixture con schema único,
saver PostgreSQL real y conexiones independientes. Cubre contador repetido/cero,
duplicados simultáneos y replay de checkpoint con un solo LLM, wait cancelable,
supersesión de invocación y stream, fencing antes/después del commit, identidad,
revocación, expiración, borrado, corte mensual, fase control y rollback no destructivo.
Los tests no llaman a ElevenLabs ni contienen datos de pacientes.

```powershell
# URI inyectada por aprovisionamiento QA verificado, BD exclusiva descartable.
uv run pytest -q tests/test_voice_native_turns_postgres.py tests/test_vita_postgres.py tests/test_vita_history_postgres.py tests/test_vita_voice.py
uv run ruff check .
```

La aceptación HTTP/audio con los nuevos hooks corresponde a root. No se afirma
que la exportación QA identifique cuál de las dos comparaciones disparó el 409
original, ni que un cuerpo de proveedor no capturado tenga una forma determinada.

Evidencia local final sobre la base `4f04514`:

- Dirigidas de admisión, PG, historial y voz: **78 passed, 16 warnings in 17.39s**.
- Suite completa, con `VITA_TEST_DATABASE_URL` de BD propia verificada en QA 55437:
  **770 passed, 47 warnings in 58.50s**, sin omitidas.
- `uv run ruff check .`: correcto. Formato: cinco archivos Python correctos.
- `git diff --check`: correcto. Alembic: un head, `94e1c2b75a06`.
- Avisos: Starlette/httpx, `FakeDbSession.add` de fixtures anteriores y
  `path_separator` de Alembic. Ningún test usa un proveedor o paciente real.
