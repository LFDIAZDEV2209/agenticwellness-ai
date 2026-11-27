# Piloto privado Vita · tocar para hablar (KAN-165)

La app abre el micrófono después del consentimiento, graba hasta 20 s y envía
solo cuando el usuario toca «Enviar grabación». Cancelar o cerrar no envía. El
gateway comprueba JWT de app, paciente activo, consentimiento de chat y voz, y
cuota por cuenta. Solo reenvía audio acotado y el idioma al AI Service, más su
propia credencial interna. No reenvía JWT, ID de cuenta, nombre de archivo, hilo
ni contexto clínico. Si el piloto falla no abre ElevenLabs ni Web Speech.

AI Service convierte WebM/MP4/WAV en WAV mono de 16 kHz en memoria, transcribe
en un sidecar privado, aplica la seguridad de Vita antes y después de generar,
y sintetiza la respuesta con eSpeak NG. Los sidecars solo se contactan por
loopback. No se usan claves externas, herramientas, historiales ni embeddings.
No se guarda audio, transcripción o respuesta. Cuando la confianza de voz es
baja o no se detecta habla, responde con una solicitud audible de repetición
sin pedir al modelo que adivine el texto. El cliente muestra el turno efímero
y permite volver a reproducir si el navegador bloquea el autoplay.

## Preparar un entorno aislado

1. Construir la imagen AI con `--build-arg INSTALL_PRIVATE_TAP=true`. La imagen
   estándar no instala los codecs adicionales. Ejecutar como usuario no root
   con filesystem restringido y dimensionar cuota CPU/memoria y tiempo de
   espera para el conjunto. No montar un filesystem con datos de pacientes.
2. Lanzar [whisper-server](https://github.com/ggml-org/whisper.cpp/tree/master/examples/server)
   con un modelo multilingüe montado en solo lectura y bind
   `127.0.0.1:8081`. Lanzar [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
   con un modelo instruct multilingüe de artefactos aprobados, alias
   `vita-local`, y bind `127.0.0.1:8082`. Empaquetar versiones y artefactos
   fijados antes de arrancar. Los procesos deben compartir el namespace de red
   con AI Service. No publicar los puertos ni permitir egress desde los
   sidecars. No activar grabación de audio, trazas de prompts o volcados.
3. Configurar `VITA_PRIVATE_TAP_ENABLED=true` en AI Service y, si se requieren,
   `VITA_TAP_STT_PORT`, `VITA_TAP_CHAT_PORT`, `VITA_TAP_MODEL`. Configurar
   `Vita__PrivateTapVoiceEnabled=true` en el API. Solo el build de prueba
   obtiene `VITE_VITA_PRIVATE_TURN=true`. Sin esas señales todos los flujos
   actuales quedan sin cambios.
4. Ejecutar `node e2e/vita-press-to-talk.mjs` desde la app contra Vite con el
   flag. En el piloto real repetir con un iPhone/TestFlight y Android.
   Verificar ES/EN, permisos revocados, micrófono cancelado, silencio,
   ruido, latencia y calidad de voz. Capturar solo destinos, estados y
   tiempos con entrada sintética y comprobar que no hay salida externa.
5. Antes de cambiar producción, demostrar aislamiento y calidad con modelos
   reales y decidir cómo se activan las otras entradas de voz. El piloto solo
   cubre el botón de voz del panel Vita y preguntas generales sin memoria ni
   acceso a expedientes. La conversación continua y la paridad de herramientas
   no forman parte del piloto.

Cerrar KAN-165 requiere además la evidencia de las otras rutas de IA,
retención y acuerdo. Estas señales no autorizan deshabilitar la voz actual
en TestFlight ni sustituir una revisión de despliegue.


## Perfil reproducido localmente

- whisper.cpp v1.8.3 (`2eeeba56`), con
  [`ggml-small-q5_1.bin`](https://huggingface.co/ggerganov/whisper.cpp/tree/5359861c739e955e79d9a303bcbc70fb988958b1).
  Verificar SHA-1 `6fe57ddcfdd1c6b07cdcc73aaf620810ce5fc771` y usar el
  endpoint `/inference` con `verbose_json` para confianza por segmento.
- llama.cpp b12774 con
  [Qwen2.5-1.5B Instruct Q4_K_M](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/91cad51170dc346986eccefdc2dd33a9da36ead9).
  SHA-256 `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`.
  Alias `vita-local`, modo offline, una ranura y contexto de 4.096 tokens.
- Para repetir la prueba de navegador, colocar `es.webm` y `en.webm` en
  `COP_E2E_TAP_BILINGUAL_AUDIO_DIR`, y apuntar `COP_E2E_TAP_GATEWAY_URL` a
  un host local de pruebas con autenticación sintética. La prueba verifica
  la intención de una pregunta fija sobre recordar beber agua en cada idioma
  y reproduce el WAV con el reproductor real de Chromium.
- Ejecutar con solo interfaz loopback; en Linux se puede usar
  `unshare -Urn` y habilitar `lo` dentro de ese espacio de red. Este perfil
  demuestra integración, no calidad en todos los acentos ni capacidad o
  ausencia de egreso del despliegue de FYA.
