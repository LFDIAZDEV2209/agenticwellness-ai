# Auditoría del harness Vita · Ola A

El harness ejecuta guardrails, prepara el contexto, aplica el registro de herramientas y registra
una traza por turno. El grafo reactivo ofrece herramientas al modelo. La variante de respuesta
directa decide las lecturas en código y reserva el modelo para la respuesta. Ambas reutilizan
los nodos, el checkpointer y las políticas de ejecución. [Ensamblado del grafo](../app/graph/graph.py)

## Matriz por canal

| Canal y entrada | Perfil y herramientas inyectadas | Límites en AI Service | Salida y modos de fallo |
| --- | --- | --- | --- |
| App · `POST /api/v1/chat/stream` con `X-Internal-Key`. `supports_vita` y el flag del runtime activan el catálogo de paciente. | Vita resuelve un perfil publicado y autorizado o el asistente general. Inyecta `get_my_day`, `get_my_nutrition`, `get_my_assessments`, `get_my_podcast`, `get_my_exercise` y `get_my_progress`, o la lectura del ámbito detectado. RAG usa las bases de conocimiento autorizadas. El camino legado usa el perfil solicitado o el supervisor. `mark_control_declined` se usa en la app con un contexto de control activo. | Generación de Vita 20 s por defecto. Camino legado 90 s. Lecturas HTTP de Vita 3 s por defecto y clasificador opcional 2 s. | SSE `start`, `message`, `node`, `flow`, `done` y, cuando corresponde, `control_signal`. `done` transporta acciones, fuentes y referencias de ejecución. Sin identidad en el flujo Vita se rechaza con 403. Un hilo ocupado en la preparación puede devolver 409. Fallos de runtime previos al stream pueden devolver 503. Un fallo en generación emite `error`. |
| Voz · `POST /internal/voice/v1/chat/completions` con bearer de servicio, sesión firmada, conversación y contador. | Selección con el mismo catálogo de paciente que el chat Vita. Usa lecturas preparadas para ámbitos directos y RAG autorizado si aplica. En consultas de ficha, el modelo elige las tools y `answer` genera el audio con sus resultados. El cambio de idioma es un control del adaptador. | Hasta 20 s por defecto para esperar una admisión o ejecutar la generación, por separado. El proveedor tiene su propio límite de cascada. | SSE compatible con OpenAI, con `chat.completion.chunk` y `[DONE]`. Reintentos del mismo turno pueden reutilizar su respuesta. Firma, sesión o vínculo inválidos devuelven 401/403 según el control. Un conflicto de admisión o hilo puede devolver 409. Fallos de preparación o espera pueden devolver 503. Una interrupción durante generación emite un código fijo en SSE. |
| WhatsApp · el backend llama a `POST /api/v1/chat` con `agent=whatsapp`, `supports_vita=true`, `channel=whatsapp` y `channel_message_id=<wamid>`. | Con `VITA_RUNTIME_ENABLED` y `VITA_WHATSAPP_ENABLED` (por defecto activo) usa el mismo runtime Vita que la app: catálogo autorizado (elegibilidad de texto), lecturas de ficha, RAG de programa y expediente y captura confirmada. Alcance sin agendamiento y formato de mensajería por canal. Comparte el hilo y el corte mensual de la app. Sin `supports_vita` o con el flag apagado vuelve al perfil legado (`calculate`, `get_current_time`). Adjuntos por WhatsApp quedan para 6.6. | Generación de Vita 20 s por defecto; camino legado 90 s. El transporte del backend tiene su propio plazo. | JSON de `ChatResponse` con respuesta, `thread_id`, herramientas y `execution_id`. El guard de catálogo devuelve texto seguro. Fallos de proveedor o timeout conservan el error HTTP del adaptador. La entrega de la respuesta de respaldo pertenece al backend de WhatsApp. |

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
  global, el grafo cierra con una respuesta útil. [Política de tools](../app/tools/policy.py)

## Lecturas de ficha · CONTRATOS-A

Los DTOs del backend se validan en el AI Service. Cada tool devuelve
`status` y `data` para una lectura disponible, o un código y texto seguro para
un fallo. El cliente envía `X-Internal-Key` y obtiene el UUID autenticado del
turno para `userId`. El backend aplica pertenencia y consentimiento. Chat y voz
pueden usar estas lecturas cuando el turno pide datos personales.

| Tool | Recurso bajo `GET /api/v1/internal/vita` | Resultado |
| --- | --- | --- |
| `get_my_profile` | `/profile` | Perfil, edad, metas vigentes, alergias, restricciones y antropometría con historial. |
| `get_my_vitals` | `/vitals` | Última lectura por métrica e historial con fecha, unidad y origen. Admite `from` y `to` locales, hasta 31 días inclusivos. El backend resuelve los extremos omitidos. |
| `get_my_medications` | `/medications` | Tratamientos vigentes, dosis, horarios y estados de las tomas del día. |
| `get_my_nutrition_plan` | `/nutrition` | Ciclo publicado de hasta 30 días, recomendaciones visibles e historial de comidas. `date` permite consultar otra fecha local. |
| `get_my_assessment_results` | `/assessments` | Puntajes e interpretaciones autorizadas de evaluaciones completadas. Las baterías pendientes o en curso aportan estado y metadatos. |
| `list_my_documents` | `/documents` | Nombre, tipo, fecha y estado del expediente, con paginación por `page` y `pageSize`. |
| `read_my_document` | `/documents/{id}` | Extracto de texto de un documento propio, seleccionado por su ID. |

Un saludo no prepara lecturas. En voz, la selección para consultas de ficha puede
añadir una invocación al modelo. La respuesta al paciente se genera después de la
lectura. El presupuesto del harness acota los resultados largos al armar el prompt.
WhatsApp tiene la misma matriz que chat y voz (KAN-74 8.1). Siguen fuera de WhatsApp
`get_patient_data` (perfil legado) y `mark_control_declined` (controles de la app);
`suggest_appointment` sigue retirada en todos los canales.
[Registro](../app/tools/vita.py) · [DTOs](../app/vita/contracts.py)

El cliente distingue perfil inactivo, acceso de medicación deshabilitado y
consentimiento pendiente (403), módulo apagado (404), consentimiento no verificable
(503) y timeout. La cancelación del turno interrumpe la lectura HTTP. Los detalles
del backend quedan fuera de la respuesta de error y de las trazas técnicas.
[Cliente HTTP](../app/vita/client.py) · [Pruebas de contrato y runtime](../tests/test_vita_record_tools.py)

Para documentos, la firma de 120 s se consume dentro del servicio y queda fuera del
resultado de la tool. Se admiten enlaces relativos a `/api/v1/storage/` del backend
y el origen HTTPS de `VITA_DOCUMENT_STORAGE_ORIGIN`, que por defecto apunta al
bucket de producción. La descarga va sin `X-Internal-Key` y sin redirecciones.
La descarga y la extracción tienen un plazo conjunto con la consulta de metadatos.
El proceso se detiene al cancelar el turno, limita el archivo a 5 MiB y devuelve
hasta 6.000 caracteres. Admite texto y PDF con texto extraíble. Un documento
escaneado o un formato sin extracción devuelve un mensaje de indisponibilidad.
En voz, el modelo puede listar y luego leer en el mismo turno antes de hablar.
[Descarga y extracción](../app/vita/document_download.py) · [Pruebas de ausencia y descarga](../tests/test_vita_extended_reads.py)

La suite de aislamiento usa dos identidades con el mismo ID de hilo visible,
lecturas distintas y memoria y trazas en PostgreSQL. Incluye una prueba de
control que omite el filtro de propietario de memoria y comprueba que la
aserción detecta la fuga. [Pruebas de aislamiento](../tests/test_vita_patient_isolation_postgres.py)

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

## Captura confirmada · KAN-70

Las tools `record_measurement`, `record_medication_dose`, `record_meal` y `record_checkin`
solo **proponen**: llaman a `POST /api/v1/internal/vita/capture/propose` y devuelven al modelo
los valores canónicos. El token AES-GCM queda en el `artifact` del ToolMessage (checkpointer);
no forma parte del texto que ve el modelo, del payload al proveedor ni de la API de hilos.

`confirm_capture` es la única que escribe (`capture/confirm`) y lo hace solo si:

1. hay propuestas vigentes del **turno inmediatamente anterior** (nunca del mismo turno);
2. el mensaje actual del paciente es una confirmación explícita y breve, evaluada en código
   (sin cifras, negaciones ni matices; en voz se evalúa la última frase transcrita);
3. el canal coincide con el de la propuesta; un valor inusual exige `confirmed_unusual`.

Identidad y procedencia salen del turno autenticado: `userId` del `configurable`, canal
`vita_chat`/`vita_voice`/`vita_whatsapp` y `turnId` determinista (`wamid`, ID del mensaje de
voz o `execution_id`, más un resumen del contenido; SHA-256 si supera 256 caracteres). El
recibo único `(paciente, canal, turnId, tool)` de BACK hace idempotente el reintento. Un fallo
transitorio de la confirmación conserva la propuesta para que el siguiente «sí» reintente con
el mismo token y `turnId`.

Un reporte de dato (`capture_query`) o la respuesta a una propuesta pendiente enrutan el turno
al planificador con tools en los tres canales; en voz, `answer` genera el audio tras las tools.
Las tools solo las concede el overlay Vita: una config ERP que las liste no las activa.
Flags: `VITA_CAPTURE_ENABLED` en AI y `Vita:CaptureEnabled` en BACK (apagado, la propuesta
responde 404 y la tool informa que no está disponible).

Límite conocido: si la llamada se corta justo durante `confirm_capture`, el turno interrumpido
se descarta y la propuesta deja de estar pendiente; el paciente debe repetir el dato y BACK
recibe un `turnId` nuevo.
[Contrato](../app/vita/capture.py) · [Tools](../app/tools/vita_capture.py) ·
[Pruebas](../tests/test_vita_capture.py)
