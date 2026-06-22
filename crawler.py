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

# WICHTIG: Erwartetes JSON-Format auf ein Objekt geändert, damit Groqs Validator nicht abstürzt.
# Die Nutzer-Fehler wurden auf ein realistisches Niveau balanciert.
prompt = f"""
Du bist ein Daten-Generator für ein neues KI-Modell. Generiere exakt 2 separate, unterschiedliche Chat-Protokolle auf Deutsch zum Thema: "{thema}".

WICHTIGE REGELN FÜR REALISTISCHE NUTZER-EINGABEN:
1. Der NUTZER ('user') schreibt kurz, umgangssprachlich und faul – wie echte Menschen im Chat (z.B. Kleinschreibung, ab und zu Wörter wie 'idk', 'kp', 'vllt', 'safe', 'kein bock', 'zocken'). Die Sätze müssen aber grammatikalisch Sinn ergeben und lesbar sein! Keine extremen, unnatürlichen Grammatikfehler.
2. Der NUTZER ('user') benutzt absolut KEINE Emojis. Seine Nachrichten enthalten nur Text!
3. Der ASSISTANT ('assistant') antwortet ebenfalls locker, freundlich und im 'Du'-Stil, bleibt aber verständlich und kurz (maximal 2-5 Sätze pro Antwort).
4. Der ASSISTANT ('assistant') nutzt VIELE Emojis (😊, 🤔, 🤷‍♂️, 😂) sinnvoll in fast jedem Satz, passend zum lockeren Ton.

AUSGABEFORMAT:
Du MUSST mit einem validen JSON-Objekt antworten, das den Key "chats" enthält. Keine Markdown-Blöcke!
{{
  "chats": [
    [
      {{"role": "user", "content": "hi wie gehts voll langweilig gerade"}},
      {{"role": "assistant", "content": "Hey! 😊 Oh nein, Langeweile ist fies. Bock auf ne Runde zocken? 🎮"}}
    ],
    [
      {{"role": "user", "content": "morgen bin voll müde kp warum"}},
      {{"role": "assistant", "content": "Guten Morgen! ☕ Oh je, das kenne ich. Schnapp dir erst mal einen Kaffee! 😊"}}
    ]
  ]
}} Das ist nur ein Beispiel die Längen der Chats sollen jeweils 10 solche einträge erhalten(nicht nur 2 wie im Bspl.)
"""

print(f"Generiere 10 Daten-Batches für Thema: {thema}")

# ==========================
# Groq Anfrage (Optimiert für gpt-oss-20b Batching)
# ==========================

completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    temperature=0.65,  # Etwas niedriger, damit das Modell strikt beim JSON-Format bleibt
    max_tokens=4096,   # Erhöht, damit alle 10 Verläufe ohne Abschneiden Platz finden
    response_format={"type": "json_object"},  # Erzwingt die korrekte JSON-Ausgabe des Modells
    messages=[
        {
            "role": "system",
            "content": "Du bist ein präziser Batch-Daten-Generator. Du antwortest ausschließlich mit einem validen JSON-Objekt, das eine Liste von exakt 10 Chat-Arrays unter dem Key 'chats' enthält."
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
# JSON prüfen und extrahieren
# ==========================

try:
    daten_objekt = json.loads(antwort)
    
    # Hier ziehen wir das Array aus dem "chats"-Key heraus, damit dein altes Datenformat exakt gleich bleibt
    if isinstance(daten_objekt, dict) and "chats" in daten_objekt:
        neue_daten_batch = daten_objekt["chats"]
    elif isinstance(daten_objekt, dict):
        # Fallback, falls der Key anders heißt
        neue_daten_batch = list(daten_objekt.values())[0]
    else:
        neue_daten_batch = daten_objekt
        
except Exception as e:
    print("Fehler beim Parsen des JSON von der KI. Inhalt war:")
    print(antwort)
    raise e

system_prompt = {
    "role": "system",
    "content": "Du bist Gleneration, eine freundliche KI. ✨"
}

# Verarbeitet alle 10 Verläufe und fügt jeweils den System-Prompt vorne an
neue_konversationen = []
for verlauf in neue_daten_batch:
    if isinstance(verlauf, list):
        konversation = [system_prompt] + verlauf
        neue_konversationen.append(konversation)

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
# Neue Daten anhängen (Nutzt .extend() für Listen)
# ==========================

gesamtes_wissen.extend(neue_konversationen)

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

print("wissen.json erfolgreich mit 10 Verläufen aktualisiert!")










