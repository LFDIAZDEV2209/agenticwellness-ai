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
No se guarda audio, transcripción o respuesta. El cliente muestra el turno
efímero y permite volver a reproducir si el navegador bloquea el autoplay.

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
