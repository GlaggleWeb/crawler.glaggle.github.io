import os
import json
import random
import tempfile
import time

from groq import Groq
from huggingface_hub import HfApi, hf_hub_download, login

# ==========================
# API-Keys
# ==========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not GROQ_API_KEY or not HF_TOKEN:
    raise ValueError("GROQ_API_KEY oder HF_TOKEN fehlt!")

login(token=HF_TOKEN)

client = Groq(api_key=GROQ_API_KEY)
api = HfApi()

REPO_ID = "GlaggleWeb/glenerationwissen"
DATEI_NAME = "wissen.json"

# ==========================
# Thema auswählen
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
print(f"🎬 Starte neutrale Text-Generierung für Thema: {thema}")

# ==========================
# Vorhandene Datei laden
# ==========================
gesamtes_wissen = []
try:
    datei = hf_hub_download(repo_id=REPO_ID, filename=DATEI_NAME, repo_type="dataset")
    with open(datei, "r", encoding="utf-8") as f:
        gesamtes_wissen = json.load(f)
    print(f"📦 Vorhandene Einträge: {len(gesamtes_wissen)}")
except Exception as e:
    print("Datei existiert noch nicht oder Fehler beim Laden. Starte neu.")

system_prompt = {
    "role": "system",
    "content": "Du bist Gleneration, eine freundliche KI. ✨"
}

# ==========================
# Schleife (5 Durchläufe à 2 Chats = 10 Chats gesamt)
# ==========================
for durchlauf in range(1, 6):
    print(f"🔄 Batch {durchlauf}/5 wird generiert...")

    prompt = f"""
Du bist ein präziser Daten-Generator für reines Textmaterial. Generiere exakt 2 separate, unterschiedliche Chat-Protokolle auf Deutsch zum Thema: "{thema}".
Jeder Chat MUSS exakt 10 Nachrichten lang sein (5x User, 5x Assistant abwechselnd).

REGELN FÜR DIE TEXTE:
1. NUTZER ('user'): Schreibt kurz, umgangssprachlich, durchgehend klein (z.B. 'idk', 'kp', 'vllt', 'safe', 'zocken'). Absolute Emojis-Sperre!
2. ASSISTANT ('assistant'): Antwortet locker, freundlich im 'Du'-Stil. Nutzt VIELE Emojis (😊, 🤔, 😂) in jedem Satz. Sätze müssen normales, sauberes Deutsch sein (ca. 2-4 Sätze pro Antwort). Keine fiktiven Links oder Platzhalter (wie 'von X').

AUSGABEFORMAT:
Antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt mit dem Key "chats". Kein Markdown, kein Text davor oder danach!
{{
  "chats": [
    [
      {{"role": "user", "content": "hi wie gehts voll langweilig gerade"}},
      {{"role": "assistant", "content": "Hey! 😊 Oh nein, Langeweile ist fies. 😩 Bock auf ne Runde zocken? 🎮 Ich bin am Start! ✨"}}
    ],
    [
      {{"role": "user", "content": "morgen bin voll müde kp warum"}},
      {{"role": "assistant", "content": "Guten Morgen! ☕ Oh je, das kenne ich gut. 🥱 Schnapp dir erst mal einen großen Kaffee! 😊 Das hilft immer. ✨"}}
    ]
  ]
}}
"""

    try:
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",       # Ein absolut stabiles, Llama-freies Modell auf Groq
            temperature=0.5,                  # Niedriger Wert für strikte Einhaltung der JSON-Struktur
            max_tokens=3000,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Du antwortest ausschließlich mit einem validen JSON-Objekt, das eine Liste von exakt 2 Chat-Arrays unter dem Key 'chats' enthält. Halte dich strikt an die Syntax."
                },
                {"role": "user", "content": prompt}
            ]
        )

        antwort = completion.choices[0].message.content.strip()
        daten_objekt = json.loads(antwort)
        neue_daten_batch = daten_objekt.get("chats", [])

        for verlauf in neue_daten_batch:
            if isinstance(verlauf, list) and len(verlauf) == 10:
                konversation = [system_prompt] + verlauf
                gesamtes_wissen.append(konversation)
        
        time.sleep(1)

    except Exception as e:
        print(f"⚠️ Fehler in Batch {durchlauf}, wird übersprungen... ({e})")
        continue

# ==========================
# Speichern und Upload
# ==========================
print(f"💾 Neuer Gesamtbestand: {len(gesamtes_wissen)} Konversationen.")

with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as temp:
    json.dump(gesamtes_wissen, temp, ensure_ascii=False, indent=2)
    temp_path = temp.name

api.upload_file(
    path_or_fileobj=temp_path,
    path_in_repo=DATEI_NAME,
    repo_id=REPO_ID,
    repo_type="dataset"
)

print("🎉 wissen.json erfolgreich und absolut sauber auf Hugging Face aktualisiert!")
if os.path.exists(temp_path):
    os.remove(temp_path)
