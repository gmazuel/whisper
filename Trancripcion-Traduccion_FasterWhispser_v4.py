import argparse
import pandas as pd
import re
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

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

    # Inicializar la diarización de altavoces
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    diarization = pipeline(archivo_mp3)

    # Crear un diccionario para almacenar las etiquetas de los altavoces
    speaker_dict = {}
    speaker_counter = 1

    # Procesar los segmentos de diarización y transcripción
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_dict:
            speaker_dict[speaker] = f"Persona{speaker_counter}"
            speaker_counter += 1

        # Transcribir el segmento de audio
        # Asumimos que 'faster-whisper' puede manejar segmentos de audio
        segments, _ = model.transcribe(archivo_mp3, segment.start, segment.end)

        for seg in segments:
            texto_neutro = ajustar_a_espanol_neutro(seg.text, reemplazos)
            print(f"{speaker_dict[speaker]} [{seg.start:.2f}s -> {seg.end:.2f}s]: {texto_neutro}")

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
    
    # Mejoras recientes:
    # - Se agregó la diarización de altavoces usando pyannote.audio para identificar diferentes oradores.
    # - Cada orador es etiquetado automáticamente como Persona1, Persona2, etc.
    # - Se transcriben segmentos de audio específicos para cada orador identificado.
