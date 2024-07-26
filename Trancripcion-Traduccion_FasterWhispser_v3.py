import argparse
import pandas as pd
import re
from faster_whisper import WhisperModel

def cargar_reemplazos(archivo_csv):
    reemplazos = {}
    df = pd.read_csv(archivo_csv)
    for _, row in df.iterrows():
        reemplazos[row['chileno']] = row['espanol_neutro']
    return reemplazos

def ajustar_a_espanol_neutro(texto, reemplazos):
    for chileno, neutro in reemplazos.items():
        # Usar expresiones regulares para reemplazar solo palabras completas
        texto = re.sub(rf'\b{re.escape(chileno)}\b', f'{chileno}->{neutro}', texto)
    return texto

def transcribir_audio(archivo_mp3, archivo_csv, modelo):
    reemplazos = cargar_reemplazos(archivo_csv)
    model = WhisperModel(modelo)
    segments, info = model.transcribe(archivo_mp3)
    for segment in segments:
        texto_neutro = ajustar_a_espanol_neutro(segment.text, reemplazos)
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {texto_neutro}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribir un archivo MP3 y transformar términos chilenos a español neutro")
    parser.add_argument("archivo_mp3", type=str, help="Ruta al archivo MP3 que se desea transcribir")
    parser.add_argument("archivo_csv", type=str, help="Ruta al archivo CSV con términos chilenos y su equivalente en español neutro")
    parser.add_argument("--modelo", type=str, default="base", help="Modelo de Whisper a utilizar: tiny, base, small, medium, large")
    args = parser.parse_args()

    transcribir_audio(args.archivo_mp3, args.archivo_csv, args.modelo)

    # Ejemplo de uso:
    # python script.py audio.mp3 terminos.csv --modelo large
    # Esto utilizará el modelo "large" de Whisper para la transcripción.
    # Modelos disponibles:
    # - tiny: Modelo más pequeño y rápido, pero menos preciso.
    # - base: Modelo básico con un buen equilibrio entre velocidad y precisión.
    # - small: Modelo más grande que el base, con mayor precisión.
    # - medium: Modelo aún más grande, con mejor precisión.
    # - large: Modelo más grande y preciso, pero más intensivo en recursos.
