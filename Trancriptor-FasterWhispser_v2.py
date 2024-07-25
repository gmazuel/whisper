import argparse
from faster_whisper import WhisperModel

def transcribir_audio(archivo_mp3):
    model = WhisperModel("base")
    segments, info = model.transcribe(archivo_mp3)
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribir un archivo MP3 usando faster-whisper")
    parser.add_argument("archivo_mp3", type=str, help="Ruta al archivo MP3 que se desea transcribir")
    args = parser.parse_args()

    transcribir_audio(args.archivo_mp3)

