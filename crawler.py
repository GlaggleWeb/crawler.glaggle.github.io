import os
import random
import json
from groq import Groq
from huggingface_hub import HfApi, login

# API-Keys laden
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Bei Hugging Face anmelden
login(token=HF_TOKEN)

client = Groq(api_key=GROQ_API_KEY)

# Hugging Face Repo-Details
REPO_ID = "dein-username/dein-repo-name"  # Anpassen!
DATEI_NAME = "wissen.json"

# Themen für vielfältige Trainings-Daten
themen = [
    "Smalltalk, Begrüßungen und wie der Tag war",
    "Über Hobbys reden (Sport, Gaming, Musik, Kochen)",
    "Einfache Alltagsfragen (Was soll ich kochen? Welcher Film ist gut?)",
    "Lustige Witze erzählen",
    "Darüber reden was der Nutzer am Wochenende machen könnte",
    "Aufmunterung und nette Worte für den Nutzer"
]

gewaehltes_thema = random.choice(themen)

# Prompt für allgemeine, nicht spezifische Beispiele
prompt = f"""
Du bist ein Daten-Generator für ein neues KI-Modell.
Generiere ein lockeres, natürliches Chat-Protokoll auf Deutsch zum Thema: "{gewaehltes_thema}".

WICHTIGE REGELN:
1. Die Fragen und Antworten müssen ALLGEMEIN und ZEITLOS sein (keine aktuellen Events, Serien, Kino)
2. Nutze SEHR VIELE Emojis (?, ?, ?, ?, etc.) sinnvoll in fast jedem Satz
3. Der Chat soll natürlich und locker wirken, wie echte Menschen schreiben
4. Gib NUR ein valides JSON-Array aus, KEINE zusätzlichen Erklärungen:

[
  {{"role": "user", "content": "Hier steht eine alltägliche Frage des Nutzers"}},
  {{"role": "assistant", "content": "Hier steht die passende Antwort mit Emojis"}}
]
"""

print(f"Generiere Alltags-Chats für: {gewaehltes_thema}...")

completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": "Du bist ein präziser Daten-Generator, der ausschließlich sauberes JSON ausgibt."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.85
)

antwort_text = completion.choices[0].message.content.strip()

try:
    # JSON parsen
    neue_konversation = json.loads(antwort_text)
    
    # System-Nachricht hinzufügen
    system_nachricht = {"role": "system", "content": "Du bist Gleneration, eine super freundliche Chat-KI! ✨"}
    komplette_nachrichten = [system_nachricht] + neue_konversation
    
    # Hugging Face API initialisieren
    api = HfApi()
    
    # Bestehende wissen.json von Hugging Face herunterladen
    print(f"Lade {DATEI_NAME} von Hugging Face herunter...")
    try:
        datei_inhalt
