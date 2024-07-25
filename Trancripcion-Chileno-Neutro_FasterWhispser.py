import argparse
import pandas as pd
from faster_whisper import WhisperModel

def cargar_reemplazos(archivo_csv):
    reemplazos = {}
    df = pd.read_csv(archivo_csv)
    for _, row in df.iterrows():
        reemplazos[row['chileno']] = row['espanol_neutro']
    return reemplazos

def ajustar_a_espanol_neutro(texto, reemplazos):
    for chileno, neutro in reemplazos.items():
        texto = texto.replace(chileno, neutro)
    return texto

def transcribir_audio(archivo_mp3, archivo_csv):
    reemplazos = cargar_reemplazos(archivo_csv)
    model = WhisperModel("base")
    segments, info = model.transcribe(archivo_mp3)
    for segment in segments:
        texto_neutro = ajustar_a_espanol_neutro(segment.text, reemplazos)
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {texto_neutro}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribir un archivo MP3 y transformar términos chilenos a español neutro")
    parser.add_argument("archivo_mp3", type=str, help="Ruta al archivo MP3 que se desea transcribir")
    parser.add_argument("archivo_csv", type=str, help="Ruta al archivo CSV con términos chilenos y su equivalente en español neutro")
    args = parser.parse_args()

    transcribir_audio(args.archivo_mp3, args.archivo_csv)

