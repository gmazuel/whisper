import csv

def cargar_reemplazos(archivo_csv):
    reemplazos = {}
    with open(archivo_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            reemplazos[row['chileno']] = row['neutro']
    return reemplazos

def ajustar_a_espanol_neutro(texto, reemplazos):
    for chileno, neutro in reemplazos.items():
        texto = texto.replace(chileno, neutro)
    return texto

# Cargar los reemplazos desde el archivo CSV
reemplazos = cargar_reemplazos('reemplazos.csv')

# Ejemplo de uso con el resultado de la transcripción
for segment in segments:
    texto_neutro = ajustar_a_espanol_neutro(segment.text, reemplazos)
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {texto_neutro}")

