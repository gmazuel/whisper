import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel

# Configuración del modelo y parámetros de audio
model_name = "base"  # Puedes elegir entre "tiny", "base", "small", "medium", "large" según tus necesidades
language = "en"  # Idioma del modelo, cambia según el idioma de tu audio
sampling_rate = 16000  # Tasa de muestreo estándar para Whisper

# Función para verificar el dispositivo utilizado
def check_device(model):
    device = next(model.parameters()).device
    if device.type == "cuda":
        print("El modelo está utilizando la GPU.")
    else:
        print("El modelo está utilizando la CPU.")

# Inicializar el modelo
model = WhisperModel(model_name, device="cpu")

# Verificar el dispositivo utilizado
check_device(model)

def transcribe_audio(audio_chunk):
    """
    Transcribe a chunk of audio data.
    """
    audio = np.frombuffer(audio_chunk, dtype=np.float32)
    audio = torch.from_numpy(audio).unsqueeze(0)
    
    # Realiza la transcripción
    result = model.transcribe(audio, language=language)
    print("Transcription:", result["text"])

def audio_callback(indata, frames, time, status):
    """
    Callback function to handle audio stream data.
    """
    if status:
        print("Status:", status, file=sys.stderr)
    if len(indata) > 0:
        # Convert the audio chunk to bytes and transcribe
        audio_chunk = indata.tobytes()
        transcribe_audio(audio_chunk)

# Configuración del flujo de audio
with sd.InputStream(callback=audio_callback, channels=1, samplerate=sampling_rate):
    print("Recording...")
    while True:
        # Keep the script running to capture audio
        try:
            sd.sleep(1000)
        except KeyboardInterrupt:
            print("Stopped.")
            break
