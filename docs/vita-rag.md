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
con un grafo cacheado. Para el expediente, `set_patient_document_status`
actualiza por paciente y documento. Tras confirmar esa transacción, el
siguiente turno excluye los chunks retirados. La ingesta y la conexión de
los eventos reales de expediente corresponden a 5.2.

En PostgreSQL 18 desechable se ejecutaron `upgrade head`, `downgrade -1` y
`upgrade head`. Con tres filas sintéticas, el retroceso conservó solo el chunk
activo del programa. Elimina los chunks privados y los inactivos para que el
runtime previo no pueda recuperarlos. La reaplicación restauró ambos índices.
La migración reconstruye HNSW y puede bloquear escrituras durante ese paso.
[Migración](../alembic/versions/67d9e4b31c52_vita_patient_rag.py)

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
