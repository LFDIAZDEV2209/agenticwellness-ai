# KAN-165 · Límite de privacidad del proveedor

Estado: entrega parcial para revisión. No es una certificación de privacidad
ni autoriza cambios en producción. El criterio de Jira exige inspeccionar lo
que sale de FYA TECH, no solo el JSON que la app envía al gateway.

## Rutas verificadas y brechas

| Ruta | Límite actual | Verificación pendiente |
| --- | --- | --- |
| Voz Vita y legacy | Antares obtiene una URL firmada del backend. La versión desplegada abre el WebSocket de ElevenLabs desde el cliente. El Custom LLM es un callback posterior de ElevenLabs al backend y no pasa por él el micrófono. | El candidato de app solo permite WebSocket del gateway. No hay un transporte de audio FYA listo. No enviar audio bruto al proveedor con un relay transparente: puede incluir identificadores y la voz de la persona. |
| Chat y clasificación | El gateway y AI Service conservan el acceso a las credenciales. `app/graph/nodes.py` envía historial, resultados de herramientas y contexto al modelo. `app/vita/runtime.py` puede enviar la pregunta original al clasificador. | Capturar el JSON real en un proveedor sintético. Minimizar campos estructurados por allowlist y mantener el texto sin procesar dentro de FYA. Una instrucción al LLM y el validador de respuesta no controlan lo enviado. |
| Adjuntos y RAG privado | `app/vita/attachment_context.py` envía imágenes como base64 al OCR. Quitar metadatos no quita nombres visibles. `app/rag/embeddings.py` puede enviar textos y consultas a OpenAI. | Ejecutar OCR/extracción y embeddings privados dentro del perímetro, o impedir salida. Incluir PDFs escaneados, fotos de documentos y resultados de herramientas en las pruebas. |
| Otros caminos de IA transversal | Food AI tiene rutas separadas y fallback configurable a Anthropic/OpenAI. La generación de planes y análisis de evaluaciones tienen endpoints internos separados. | Inventariar configuración desplegada y probar cada destino/modelo de fallback. Una prueba del chat no cubre estas rutas. |
| Logs técnicos | El candidato quita cuenta, hilo, conversación y correlación externa de las líneas del chat y de emisión legacy. La métrica de turno no imprime el ID de ejecución. | Revisar access logs, APM, gateways y excepciones en operación. Las tablas de ejecución y trazas internas aún tienen referencias que pueden volver a identificar a alguien. |

## Separación de datos y retención

- No duplicar texto bruto en logs operativos. El historial autorizado es un
  almacén distinto y ya tiene [corte mensual](vita-history.md). Los
  [adjuntos efímeros](vita-attachments.md) tienen TTL de 24 h por defecto.
  Ninguno de esos controles prueba el borrado de trazas SQL, exportaciones,
  almacenamiento externo ni respaldos.
- Aplicar [la decisión de KAN-170](https://diomedescerda.atlassian.net/browse/KAN-170?focusedCommentId=10471)
  según categoría. Logs operativos ordinarios: 30 días. RDS automático: 7 días.
  Inventariar snapshots manuales y versiones no actuales de S3 antes de aplicar
  el valor por defecto de 30 días. La eliminación elegible en sistemas activos
  tiene como máximo 90 días desde la verificación de la solicitud, sujeto a un
  plazo anterior aplicable. No es el TTL predeterminado de un prompt.
- Para el registro restringido de prompts y respuestas, definir qué versión
  minimizada se conserva y con qué TTL. Usar seudónimo con clave privada
  separada, no un hash sin clave ni el ID de sesión del proveedor. Evitar que
  las métricas públicas compartan ese seudónimo. Probar aislamiento, caducidad
  y borrado a partir de la clave de sujeto en el almacén autorizado.
- La definición de despliegue de AI usa `us-east-2`; no se verificó la
  ubicación de cada almacén ni sus políticas efectivas. Comprobar grupo de
  logs ECS y su `retentionInDays`, almacenamiento de DB/adjuntos y réplicas,
  snapshots y versiones S3. Restaurar un conjunto sintético, reconciliar las
  supresiones antes de reabrirlo y adjuntar un resultado sin datos de pacientes.

## Evidencia de cierre

1. Aprobación legal: enlace restringido al acuerdo firmado y a su alcance por
   producto, cuenta, proyecto/modelo y región. Confirmar uso permitido, no
   entrenamiento, retención/eligibilidad, copias, eliminación y subencargados.
   Adjuntar la configuración efectiva, sin claves. La documentación pública
   [de ElevenLabs](https://elevenlabs.io/docs/eleven-agents/customization/privacy/zrm)
   describe una opción y no demuestra que esté habilitada ni que el contrato
   esté firmado.
2. QA en entorno aislado, con proveedor capturador y datos ficticios en ES/EN.
   Incluir nombre, correo, teléfono y dirección, IDs de usuario/hilo y
   contenido clínico incrustado en texto, historial, visión, RAG y tools.
   Comprobar cuerpo, cabeceras, parámetros y destino después de serializar,
   tanto en éxito como en fallback, reintento y error. No almacenar capturas
   reales. Si falla la minimización, no reenviar el original.
3. Confirmar que el cliente no contacta a proveedores ni contiene credenciales
   en el artefacto firmado. Probar una versión antigua del cliente cuando el
   servidor deja de emitir sesiones directas. `VOICE_ENABLED=false` bloquea
   nuevas sesiones legacy y Vita; revisar y expirar sesiones previas antes de
   declarar cerrado ese camino. Probar un turno privado en un dispositivo
   si se restablece voz, o validar el fallback a texto si se aprueba el retiro.
4. Adjuntar evidencias de retención, permisos, expiración, restauración y
   borrado de una cuenta sintética. Comprobar que no queden copias accesibles
   por restauración ni identificadores de enlace en los logs técnicos. Dejar
   KAN-165 abierto hasta verificar los puntos anteriores.
