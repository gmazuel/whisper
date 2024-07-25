from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cpu")

def transcribir_audio_en_tiempo_real(audio_stream):
    segments, info = model.transcribe(audio_stream)
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

# Ejemplo de uso con un flujo de audio en tiempo real
audio_stream = "ruta/al/flujo/audio"
transcribir_audio_en_tiempo_real(audio_stream)

