# Adjuntos efímeros de Vita (Ola B, KAN-71: 6.1, 6.2, 6.3, 6.5)

Imágenes y PDF que el paciente adjunta al chat Vita. Por defecto son **efímeros**: no entran al
expediente (`app.documents`) ni al RAG privado (`ai.knowledge_chunks`). El archivo en expediente
con confirmación (6.4) lo implementa el backend consumiendo `GET .../content`.

- Flag: `VITA_ATTACHMENTS_ENABLED` (apagado por defecto en todos los entornos). Apagado → `403
vita_attachments_disabled` en subida y en chat con `attachment_ids`.
- Canal: interno backend → AI Service con `X-Internal-Key`. El backend autentica al paciente con
  su JWT y envía `userId`/`patientId` de esa sesión; el modelo nunca elige paciente.
- Almacén: tabla `ai.vita_attachments` (migración `b3f8d2a61c07`), binario `bytea` + metadatos
  mínimos. No guarda nombre original ni texto extraído.

## Límites y TTL (variables de entorno)

| Variable                             | Defecto      | Uso                                            |
| ------------------------------------ | ------------ | ---------------------------------------------- |
| `VITA_ATTACHMENT_MAX_IMAGE_BYTES`    | 5 MiB        | Tope imagen (Anthropic rechaza > 5 MiB)        |
| `VITA_ATTACHMENT_MAX_PDF_BYTES`      | 10 MiB       | Tope PDF                                       |
| `VITA_ATTACHMENT_MAX_PDF_PAGES`      | 20           | Páginas máximas                                |
| `VITA_ATTACHMENT_TTL_SECONDS`        | 86400 (24 h) | Vida del binario; expirado = ilegible          |
| `VITA_ATTACHMENT_PURGE_INTERVAL`     | 300 s        | Worker de purga (idempotente entre réplicas)   |
| `VITA_ATTACHMENT_MAX_TEXT_CHARS`     | 24000        | Texto extraído máximo por turno (se reparte)   |
| `VITA_ATTACHMENT_MAX_VISION_IMAGES`  | 4            | Imágenes de un PDF escaneado enviadas a visión |
| `VITA_ATTACHMENT_EXTRACTION_TIMEOUT` | 25 s         | Tope por adjunto; excedido = `unavailable`     |

## Endpoints

Base: `/internal/vita/attachments`. Todos exigen `X-Internal-Key` (`401` si falta). Errores con
cuerpo `{"detail": "<código>", "message": "<texto paciente>"?}`; nunca devuelven eco del cuerpo
recibido. Fallo de BD → `503 attachment_store_unavailable`.

### `POST /internal/vita/attachments` (multipart/form-data)

| Campo       | Tipo         | Obligatorio | Notas                                          |
| ----------- | ------------ | ----------- | ---------------------------------------------- |
| `file`      | archivo      | sí          | JPG, PNG, WEBP o PDF                           |
| `userId`    | string 1-128 | sí          | Usuario autenticado                            |
| `patientId` | UUID         | sí          | Paciente de la sesión                          |
| `threadId`  | string ≤128  | no          | thread_id del cliente (sin prefijo de usuario) |
| `language`  | `es`/`en`    | no          | Idioma de `message` en rechazos                |

`201`:

```json
{
  "attachmentId": "uuid",
  "kind": "image | pdf",
  "contentType": "image/jpeg | image/png | image/webp | application/pdf",
  "sizeBytes": 123456,
  "pageCount": 3,
  "sha256": "hex64",
  "threadId": "thread-1",
  "createdAt": "2026-11-18T12:00:00Z",
  "expiresAt": "2026-11-19T12:00:00Z"
}
```

Validación: firma real (magic bytes), coherencia con `Content-Type` declarado y extensión,
tamaño por tipo, estructura del PDF (pypdf), páginas y cifrado. **Un rechazo no almacena nada.**

| Status | `detail`                      | Causa                                         |
| ------ | ----------------------------- | --------------------------------------------- |
| 403    | `vita_attachments_disabled`   | Flag apagado                                  |
| 413    | `attachment_too_large`        | Supera el tope del tipo real                  |
| 415    | `attachment_type_not_allowed` | Firma no admitida                             |
| 422    | `attachment_empty`            | Archivo vacío                                 |
| 422    | `attachment_mime_mismatch`    | Firma ≠ tipo declarado o extensión            |
| 422    | `attachment_too_many_pages`   | PDF con más páginas que el máximo             |
| 422    | `attachment_pdf_encrypted`    | PDF con contraseña de apertura                |
| 422    | `attachment_unreadable`       | Archivo dañado / PDF sin páginas              |
| 422    | `invalid_attachment_request`  | Formulario inválido (faltan campos, UUID mal) |

### `GET /internal/vita/attachments/{attachmentId}?patientId=<uuid>`

`200` con los mismos metadatos. `404 vita_attachment_not_found` si es inexistente, de otro
paciente o expirado (indistinguibles).

### `GET /internal/vita/attachments/{attachmentId}/content?patientId=<uuid>`

Binario con `Content-Type` real, `Cache-Control: no-store`, `X-Content-Type-Options: nosniff`,
`Content-Disposition: attachment` y `X-Attachment-Sha256`. Mismo `404`. Es el punto de entrada
de 6.4: el backend lo descarga tras la confirmación del paciente, verifica el SHA-256, lo sube a
S3 privado, crea el registro documental y dispara la indexación privada existente (5.2). Después
puede borrar el efímero con `DELETE`.

### `DELETE /internal/vita/attachments/{attachmentId}?patientId=<uuid>`

`204` idempotente. Un adjunto ajeno no se borra y la respuesta no lo delata.

### `POST /internal/vita/attachments/purge`

`200 {"purged": n}`. Borra todos los expirados (el worker periódico hace lo mismo).

## Uso en chat

`POST /api/v1/chat` y `/api/v1/chat/stream` aceptan `attachment_ids: UUID[]` (máx. 3, se
deduplican). Requisitos: flag activo, `supports_vita=true`, `user_id` y `patient_id` de la sesión
y canal de texto (WhatsApp y legacy → `403`). Cada adjunto debe ser del mismo usuario/paciente,
vigente y del mismo hilo (o sin hilo); si no, `404 vita_attachment_not_found` y el modelo no se
invoca.

Respuesta: `attachments: [{attachment_id, kind, readable}]` (en stream, dentro del evento
`done`). `readable=false` → Vita dice que no pudo leerlo y pide un archivo más claro.

Extracción (6.2):

- PDF digital: capa de texto con pypdf (sin visión).
- PDF escaneado (< 20 caracteres de texto): imágenes JPEG/Flate RGB/gris de las páginas → modelo
  de visión configurado. JPX/CCITT/JBIG2 → `unreadable`.
- Imagen: visión directa. Respuesta `ILEGIBLE` del modelo → `unreadable`.
- Proveedor caído o timeout → `unavailable` (sin inventar valores).

## Defensa frente a prompt injection (6.5)

- El texto extraído se limpia (controles, bidi, delimitadores falsos) y cada frase con forma de
  instrucción (EN/ES: ignorar instrucciones, revelar prompt/secretos/otros pacientes, etiquetas
  de rol, órdenes dirigidas al asistente, peticiones de tool) se sustituye por una marca neutra.
- Va en un mensaje de **usuario** delimitado con nonce aleatorio por turno, justo antes de la
  pregunta vigente, más una `ATTACHMENT POLICY` de sistema que lo declara dato no confiable.
- Ni el texto ni el bloque entran al estado del grafo ni al checkpoint: solo `{id, kind}` como
  referencia mínima en la metadata del mensaje del paciente.
- Intento detectado → log `vita_attachment_injection_detected attachment=<id> patterns=<nombres>`
  (solo nombres de patrón estáticos, nunca contenido).

## Pruebas

```bash
uv run pytest tests/test_vita_attachments.py            # unitarias, sin BD
# PostgreSQL desechable propio (pgvector), nunca la BD compartida:
docker run -d --rm --name vita-attach-pg-$RANDOM -e POSTGRES_USER=test \
  -e POSTGRES_PASSWORD=<pw> -e POSTGRES_DB=vita_test -p 127.0.0.1:55441:5432 pgvector/pgvector:pg18
VITA_TEST_DATABASE_URL=postgresql+psycopg://test:<pw>@127.0.0.1:55441/vita_test \
  uv run pytest tests/test_vita_attachments_postgres.py
```
