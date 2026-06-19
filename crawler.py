import os
import random
import json
from groq import Groq
from datasets import load_dataset, Dataset

# API-Keys aus den GitHub Secrets laden
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "DEIN_HF_NAME/glenerationwissen" # 🔴 HIER DEINEN HF-NAMEN EINTRAGEN

client = Groq(api_key=GROQ_API_KEY)

# Deine Themenliste
themen = [
    "Quantenphysik verständlich erklärt", "Die Geschichte der künstlichen Intelligenz",
    "Wie funktioniert eine Blockchain?", "Spannende Fakten über das Universum",
    "Die wichtigsten Programmierkonzepte in Python", "Wie das menschliche Gehirn lernt"
]
gewaehltes_thema = random.choice(themen)

# WICHTIG: Wir zwingen das Modell, uns reines JSON im richtigen Format zu liefern
prompt = f"""
Generiere ein hochqualitatives Chat-Protokoll auf Deutsch zum Thema: "{gewaehltes_thema}".
Du MUSST die Antwort als valides JSON-Array ausgeben. Nutze exakt diese Struktur:

[
  {{"role": "user", "content": "Hier steht die Frage des Nutzers"}},
  {{'role': 'assistant', 'content': 'Hier steht deine ausführliche Antwort'}}
]

Gib NUR das JSON-Array aus. Keine Einleitung, keine Formatierungscodes (wie ```json), kein Text davor oder danach.
"""

print(f"Lasse gpt-oss-20b Daten generieren für: {gewaehltes_thema}...")

completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        # Hier nutzen wir die SYSTEM-Rolle, um der KI ihre Identität zu geben!
        {"role": "system", "content": "Du bist ein präziser Daten-Generator, der ausschließlich sauberes JSON ausgibt."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7
)

antwort_text = completion.choices[0].message.content.strip()

try:
    # Überprüfen, ob das Modell sauberes JSON geliefert hat
    neue_konversation = json.loads(antwort_text)
    
    # Wir fügen ganz vorne eine SYSTEM-Nachricht für dein Modell hinzu
    system_nachricht = {"role": "system", "content": f"Du bist Gleneration, eine schlaue KI. Du antwortest hilfreich auf Fragen zum Thema {gewaehltes_thema}."}
    komplette_nachrichten = [system_nachricht] + neue_konversation

    print("Lade bestehende Daten von Hugging Face...")
    try:
        dataset = load_dataset(DATASET_REPO, split="train", token=HF_TOKEN)
        bestehende_liste = dataset["messages"]
    except:
        bestehende_liste = []

    # Die neue Konversation der Liste hinzufügen
    bestehende_liste.append(komplette_nachrichten)

    # Dataset aktualisieren und hochladen
    # Jede Zeile in der Spalte "messages" enthält nun die komplette Liste (System -> User -> Assistant)
    neu_dataset = Dataset.from_dict({"messages": bestehende_liste})
    neu_dataset.push_to_hub(DATASET_REPO, token=HF_TOKEN)
    
    print("Erfolgreich im professionellen 'messages'-Format auf Hugging Face gespeichert!")

except Exception as e:
    print(f"Fehler beim Verarbeiten des JSON: {e}")
    print(f"Rohtext der KI war: {antwort_text}")
