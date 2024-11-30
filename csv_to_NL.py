import pandas as pd

# Load the dataset
dataset_path = 'dataset/final.csv'  # Modifica con il percorso del tuo dataset
dataset = pd.read_csv(dataset_path, delimiter=';')

# Trasforma il dataset in frasi di linguaggio naturale contestuali
natural_language_examples = []

for _, row in dataset.iterrows():
    zona = row['Zona']
    orario = row['Orario']
    volume_traffico = int(row['Volume Traffico'] * 1000)  # De-normalizing
    velocita = int(row['Velocità Media (km/h)'] * 100)  # De-normalizing
    rallentamenti = "moderato" if row['Eventuali Rallentamenti'] == 0 else "elevato"
    pm10 = int(row['PM10 (μg/m³)'] * 100)  # De-normalizing
    no2 = int(row['NO2 (μg/m³)'] * 100)  # De-normalizing
    o3 = int(row['O3 (μg/m³)'] * 100)  # De-normalizing
    aqi = int(row['AQI'] * 100)  # De-normalizing
    bus = row['Bus']

    # Crea una frase contestuale e utile
    if rallentamenti == "elevato":
        sentence = (
            f"A causa del traffico elevato nella zona di '{zona}' alle {orario}, "
            f"è consigliabile partire almeno 30 minuti prima o dopo per ridurre i tempi di viaggio. "
            f"Il volume di traffico è {volume_traffico} veicoli con una velocità media di {velocita} km/h."
        )
    else:
        sentence = (
            f"Il traffico nella zona di '{zona}' alle {orario} è moderato. "
            f"Puoi partire tranquillamente e aspettarti una velocità media di {velocita} km/h. "
            f"I livelli di inquinanti PM10: {pm10}, NO2: {no2}, O3: {o3} indicano un'aria relativamente pulita (AQI: {aqi}). "
            f"I bus disponibili sono: {bus}."
        )
    
    # Aggiungi la frase alla lista
    natural_language_examples.append(sentence)

# Salva le frasi in un file di testo
output_examples_path = 'llama_fine_tuning_examples.txt'
with open(output_examples_path, 'w') as file:
    for example in natural_language_examples:
        file.write(example + '\n')

print(f"File creato con successo: {output_examples_path}")
