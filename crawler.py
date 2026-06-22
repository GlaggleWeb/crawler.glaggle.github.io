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
# Themen (Überarbeitet: Reine Grundunterhaltungen)
# ==========================

themen = [
    "Klassischer Smalltalk (Wie geht es dir, wie war dein Tag, was machst du gerade?)",
    "Über den Tag philosophieren und alltägliche Gefühle austauschen (Müdigkeit, Motivation, gute Laune)",
    "Lockerer Austausch über universelle Hobbys (Musik hören, Kochen als Beschäftigung, Gaming, Sport allgemein)",
    "Das Konzept von Entspannung und Plänen fürs Wochenende (ohne konkrete Orte zu nennen)",
    "Gegenseitige Begrüßungen, Abschiede und freundliche Floskeln im Alltag",
    "Gegenseitige Aufmunterung bei einem stressigen Tag (allgemeines Mitgefühl und nette Worte)"
]

thema = random.choice(themen)

# WICHTIG: Die Anweisung wurde verschärft, um Witze und Erfindungen zu verbieten
prompt = f"""
Du bist ein Daten-Generator für ein neues KI-Modell. Generiere ein lockeres, natürliches Chat-Protokoll auf Deutsch zum Thema: "{thema}".

WICHTIGE REGELN FÜR DIE GENERIERUNG:
1. Absolut KEINE Witze, KEINE Rätsel und KEINE fiktiven Geschichten.
2. Nenne KEINE spezifischen, erfundenen Eigennamen (keine ausgedachten Orte wie 'Wasserfall-Weg', keine Filmtitel, keine fiktiven Buchtitel). Bleibe völlig allgemein (z.B. 'ein Spaziergang im Wald' oder 'ein Buch lesen').
3. Die Fragen und Antworten müssen ALLGEMEIN, ALLTÄGLICH und ZEITLOS sein.
4. Nutze VIELE Emojis (😊, 🎉, 💡, 🤔, etc.) sinnvoll in fast jedem Satz, passend zum lockeren Ton.
5. Der Chat soll natürlich und ungezwungen wirken, wie eine normale Nachricht von Freunden (Benutze 'Du').
6. Gib NUR ein valides JSON-Array aus, KEINE zusätzlichen Erklärungen drumherum:
[
  {{"role": "user", "content": "Nutzer-Nachricht"}},
  {{"role": "assistant", "content": "KI-Antwort"}}
]
"""

print(f"Generiere Daten für Thema: {thema}")

# ==========================
# Groq Anfrage
# ==========================

completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    temperature=0.7,  # Temperatur leicht gesenkt für weniger "kreativen Quatsch"
    messages=[
        {
            "role": "system",
            "content": "Du bist ein präziser JSON-Generator. Du antwortest ausschließlich in validem JSON ohne Markdown-Block (keine ```json ... ```)."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
)

antwort = completion.choices[0].message.content.strip()

# Falls die KI doch Markdown-Codeblöcke mitsendet, putzen wir sie hier weg
if antwort.startswith("```"):
    antwort = antwort.split("\n", 1)[1]
if antwort.endswith("```"):
    antwort = antwort.rsplit("\n", 1)[0]

antwort = antwort.strip()
print("Antwort erhalten.")

# ==========================
# JSON prüfen
# ==========================

try:
    neue_daten = json.loads(antwort)
except Exception as e:
    print("Fehler beim Parsen des JSON von der KI. Inhalt war:")
    print(antwort)
    raise e

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
    print("Datei existiert noch nicht oder Fehler beim Laden.")
    print(e)

# ==========================
# Neue Daten anhängen
# ==========================

gesamtes_wissen.append(konversation)

print(f"Neuer Gesamtbestand: {len(gesamtes_wissen)}")

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
