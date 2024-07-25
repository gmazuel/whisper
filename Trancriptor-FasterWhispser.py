from faster_whisper import WhisperModel

model = WhisperModel("base")
segments, info = model.transcribe("11_mariana_enriquez_la_vuelta_al_mundo_recuadro.mp3")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

