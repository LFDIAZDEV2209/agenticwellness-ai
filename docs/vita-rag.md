# Recuperación Vita por ámbitos

## Contrato

`retrieve_knowledge(query, scope)` acepta `program` y `patient`. El valor por
defecto es `program` para preservar los consumidores actuales.

| Ámbito | Autorización y filtro SQL |
| --- | --- |
| `program` | IDs de KBs activas globales y las permitidas del agente. El SQL exige que el chunk pertenezca a una de esas KBs y que el documento siga en estado `Listo` y su KB en `Activo`. Una selección vacía da cero resultados. |
| `patient` | En chat y voz de Vita, el UUID de paciente proviene del contexto autenticado del backend o de la sesión firmada. El SQL exige ese `patient_id` y estado `active` antes del ranking y del límite. Los argumentos publicados son únicamente `query` y `scope`. |

Cada resultado incluye título, origen, fecha y sección. El programa usa la fecha
registrada del documento en el catálogo ERP. El paciente usa `source_date` de
los metadatos, con fecha de indexación identificada como tal cuando falta la del
documento. Un fragmento necesita similitud coseno de al menos 0,20. La respuesta
distingue una búsqueda sin resultados de un fallo de recuperación. Las fuentes
del expediente se entregan al turno autorizado y quedan fuera de la telemetría
de ejecuciones. [Tool](../app/tools/retrieval.py) · [Filtro SQL](../app/rag/pgvector_store.py)

## Índice e invalidación

La revisión Alembic `67d9e4b31c52` agrega `patient_id` y `retrieval_status`.
La restricción de ámbito exige exactamente un dueño, KB o paciente. El índice
HNSW queda restringido a filas del programa. El índice de paciente usa
`(patient_id, retrieval_status)` y su ranking es exacto dentro del ámbito.

Retirar una KB o su documento surte efecto en la siguiente búsqueda, incluso
con un grafo cacheado. Para el expediente, el endpoint interno confirma el
alta, reemplazo o retiro por paciente y documento en una sola transacción.

En PostgreSQL 18 desechable se ejecutaron `upgrade head`, `downgrade -1` y
`upgrade head`. Con tres filas sintéticas, el retroceso conservó solo el chunk
activo del programa. Elimina los chunks privados y los inactivos para que el
runtime previo no pueda recuperarlos. La reaplicación restauró ambos índices.
La migración reconstruye HNSW y puede bloquear escrituras durante ese paso.
[Migración](../alembic/versions/67d9e4b31c52_vita_patient_rag.py)

## API interna de indexación de expediente (5.2)

Ambas operaciones requieren `X-Internal-Key` y se invocan desde el backend.
El propietario se valida en `app.documents.patient_id`. El backend debe
enviar el evento después de confirmar el estado clínico y derivar el paciente
del documento persistido.

### `POST /internal/vita/documents/index`

```json
{
  "patientId": "d8ff076c-db7e-4e4c-9e59-bf8dc2100dd5",
  "documentId": "14b6ca8b-7c19-4f88-b8d8-a2e007a0329e",
  "title": "Informe sintético",
  "documentDate": "2026-11-18T12:00:00Z",
  "status": "active",
  "content": "Texto extraído del informe.",
  "contentType": "text/plain"
}
```

| Campo | Contrato |
| --- | --- |
| `patientId`, `documentId` | UUID del paciente canónico y de la fila clínica del documento. |
| `title`, `documentDate` | Título de hasta 200 caracteres y fecha/hora con zona. Se guardan con las referencias. |
| `status` | `active` reemplaza todos los fragmentos privados del documento. `archived` o `deleted` los retira sin generar embeddings. |
| `content` | Texto extraído para `active`, hasta 2.000.000 de caracteres. |
| `contentBase64` | Alternativa a `content` para un PDF de hasta 5 MiB, con texto extraíble. PDFs escaneados requieren extracción previa. |
| `contentType` | Opcional. Si se indica en un alta, debe coincidir con el formato (`text/plain` o `application/pdf`). |

Un alta requiere exactamente uno de `content` o `contentBase64`. En bajas se
pueden omitir ambos. El `200` devuelve `documentId`, `status`, `chunksCreated`
y `replacedChunks`, que en una baja cuenta los fragmentos retirados. Dos
reintentos del mismo alta dejan un único conjunto de fragmentos.

### `DELETE /internal/vita/documents/{documentId}?patientId={uuid}`

Devuelve `204` sin cuerpo, incluso si el par ya quedó sin fragmentos. Si la fila
clínica ya no existe, puede retirar restos del par solicitado. Un propietario
distinto en el catálogo devuelve `404`. Ambos tipos de baja eliminan físicamente
los fragmentos privados y dejan intacto el programa, incluso ante UUID iguales.

| HTTP | Respuesta y efecto |
| --- | --- |
| `401` | Clave ausente o inválida. |
| `404` | `patient_document_not_found` para documento ajeno, o un POST sin fila clínica. |
| `409` | `patient_document_inactive` si el alta apunta a un documento archivado, borrado o en otro estado distinto de `Ready`. |
| `422` | `invalid_patient_document_request` o `patient_document_not_extractable`. |
| `503` | Servicio sin configuración de clave o indisponibilidad de BD/embeddings. Un reemplazo fallido conserva el último conjunto confirmado. |

Los errores devuelven `{"detail":"..."}` sin contenido clínico. El servicio
requiere lectura de `app.documents` y la revisión `67d9e4b31c52` ya instalada.
La extracción y los embeddings se ejecutan fuera de la transacción de escritura.
Antes de reemplazar los fragmentos, el servicio vuelve a comprobar propietario y
estado bajo un bloqueo por documento para rechazar altas demoradas tras un
archivo o borrado clínico. Mientras una fila siga `Ready`, dos altas distintas
se aplican en orden de escritura. El emisor debe mantener el orden si envía
distintas versiones del mismo UUID. El flujo del emisor .NET y el proveedor de
embeddings en producción requieren validación conjunta.
[Pruebas HTTP con PostgreSQL](../tests/test_vita_document_index_postgres.py)

## Batería ES/EN y latencia

| Cobertura | Preguntas | Resultado esperado comprobado |
| --- | ---: | --- |
| Guía del programa | 4 | Fuente global |
| Fuente de especialista | 4 | KB autorizada, incluso con un fragmento ajeno mejor puntuado |
| Documento privado de paciente A o B | 4 | Solo el expediente asignado, incluso cuando el fragmento del otro paciente tiene mayor similitud |
| Tema sin respaldo autorizado | 3 | Cero fuentes y aviso explícito |

**15 de 15 preguntas** cumplieron el resultado esperado en tres rondas.
**45 búsquedas** en PostgreSQL 18 local con vectores sintéticos medidos desde
la invocación de la tool hasta generar los extractos y el bloque de fuentes.
**p50 7,42 ms. p95 8,66 ms.** El corpus de prueba tiene siete chunks. El
perfil de latencia del proveedor de embeddings y de la llamada queda por medir.

Una prueba concurrente inició dos cálculos de embedding síncronos en hilos
distintos y comprobó que el event loop avanzó mientras ambos esperaban. Las
consultas y el ranking se ejecutan con SQL asíncrono. También se verificaron
el grafo de chat y voz con perfil base y especialista, y la invalidación por
estado. Estas pruebas usan modelos de respuesta y vectores sintéticos.
[Batería reproducible](../tests/test_vita_rag_postgres.py)

El registro de Ola 0 conserva las ráfagas HTTP 409 del 14-11. La regresión local
de Ola B reprodujo `vita_thread_busy` al colgar durante una tool y pasar a chat.
Ese resultado identifica un mecanismo adicional de colisión en el mismo hilo.
La evidencia histórica por sí sola no atribuye todos los 409 al mismo mecanismo.
[Diagnóstico de voz](vita-runtime.md) · [Regresión PostgreSQL](../tests/test_voice_native_turns_postgres.py)
