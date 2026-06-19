import os
import json
import random
import tempfile

from groq import Groq
from huggingface_hub import HfApi, hf_hub_download, login

# ==========================
# API-Keys
# ==========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY fehlt!")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN fehlt!")

login(token=HF_TOKEN)

client = Groq(api_key=GROQ_API_KEY)
api = HfApi()

# ==========================
# Hugging Face Repo
# ==========================

REPO_ID = "GlaggleWeb/glenerationwissen"
DATEI_NAME = "wissen.json"

# ==========================
# Themen
# ==========================

themen = [
    "Smalltalk und Begrüßungen",
    "Hobbys",
    "Kochen",
    "Gaming",
    "Filme",
    "Musik",
    "Witze",
    "Wochenende",
    "Freundliche Gespräche",
    "Motivation"
]

thema = random.choice(themen)

prompt = f"""
Du bist ein Daten-Generator.

Erstelle eine natürliche deutsche Unterhaltung zum Thema:

{thema}

Regeln:

- Nur Deutsch
- Locker und freundlich
- Viele Emojis
- Zeitlos
- Nur JSON ausgeben

Format:

[
  {{
    "role": "user",
    "content": "..."
  }},
  {{
    "role": "assistant",
    "content": "..."
  }}
]
"""

print(f"Generiere Daten für Thema: {thema}")

# ==========================
# Groq Anfrage
# ==========================

completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    temperature=0.85,
    messages=[
        {
            "role": "system",
            "content": "Gib ausschließlich valides JSON aus."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
)

antwort = completion.choices[0].message.content.strip()

print("Antwort erhalten.")

# ==========================
# JSON prüfen
# ==========================

neue_daten = json.loads(antwort)

system_prompt = {
    "role": "system",
    "content": "Du bist Gleneration, eine freundliche KI. ✨"
}

konversation = [system_prompt] + neue_daten

# ==========================
# Vorhandene Datei laden
# ==========================

gesamtes_wissen = []

try:
    datei = hf_hub_download(
        repo_id=REPO_ID,
        filename=DATEI_NAME,
        repo_type="dataset"
    )

    with open(datei, "r", encoding="utf-8") as f:
        gesamtes_wissen = json.load(f)

    print(f"Vorhandene Einträge: {len(gesamtes_wissen)}")

except Exception as e:
    print("Datei existiert noch nicht.")
    print(e)

# ==========================
# Neue Daten anhängen
# ==========================

gesamtes_wissen.append(konversation)

print(
    f"Neuer Gesamtbestand: {len(gesamtes_wissen)}"
)

# ==========================
# Temporäre Datei erstellen
# ==========================

with tempfile.NamedTemporaryFile(
    mode="w",
    suffix=".json",
    delete=False,
    encoding="utf-8"
) as temp:

    json.dump(
        gesamtes_wissen,
        temp,
        ensure_ascii=False,
        indent=2
    )

    temp_path = temp.name

# ==========================
# Upload
# ==========================

api.upload_file(
    path_or_fileobj=temp_path,
    path_in_repo=DATEI_NAME,
    repo_id=REPO_ID,
    repo_type="dataset"
)

print("wissen.json erfolgreich aktualisiert!")
