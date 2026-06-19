import os
import random
import json
from groq import Groq
from datasets import load_dataset, Dataset

# API-Keys aus den GitHub Secrets laden
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "GlaggleWeb/glenerationwissen" # 🔴 HIER DEINEN HF-NAMEN EINTRAGEN

client = Groq(api_key=GROQ_API_KEY)

# Eine Liste mit komplett alltäglichen Chat-Themen und Stimmungen
themen = [
    "Smalltalk, Begrüßungen und wie der Tag war",
    "Über Hobbys reden (Sport, Gaming, Musik, Kochen)",
    "Chatbeipiel ohne detailliertem Wissen mit Mensch und KI",
    "Einfache Alltagsfragen (Was soll ich kochen? Welcher Film ist gut?)",
    "Lustige Witze erzählen die sinn ergeben (nicht zusammengebastelte)",
    "Darüber reden was der Nutzer am Wochenende(oder auch andere Wochentage) machen könnte",
    "Aufmunterung und nette Worte für den Nutzer"
]
gewaehltes_thema = random.choice(themen)

# Der neue Prompt: Erzeugt allgemeine Trainingsdaten mit vielen Emojis!
prompt = f"""
Du bist ein Daten-Generator für ein neues KI-Modell, das lernen soll, wie ein Mensch zu chatten.
Generiere ein lockeres, natürliches Chat-Protokoll auf Deutsch zum Thema: "{gewaehltes_thema}".

WICHTIGE REGELN:
1. Die Fragen und Antworten müssen allgemein und alltagstauglich sein, damit ein neues Modell das freie Chatten lernt.
2. Nutze SEHR VIELE Emojis (🌟, 🚀, 😊, 🎉, 🤔 etc.) in fast jedem Satz, um den Chat lebendig zu machen! Aber setze sie Sinnvoll ein.
3. Du MUSST die Antwort als valides JSON-Array ausgeben. Nutze exakt diese Struktur:

[
  {{"role": "user", "content": "Hier steht eine alltägliche Frage oder Nachricht des Nutzers"}},
  {{"role": "assistant", "content": "Hier steht die passende, freundliche Antwort mit vielen Emojis"}}
]

Gib NUR das JSON-Array aus. Keine Einleitung, keine Formatierungscodes (wie ```json), kein Text davor oder danach.
"""

print(f"Lasse gpt-oss-20b Alltags-Chats generieren für: {gewaehltes_thema}...")

completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": "Du bist ein präziser Daten-Generator für KI-Training, der ausschließlich sauberes JSON ausgibt."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.85 # Etwas höhere Kreativität für abwechslungsreichere Chats
)

antwort_text = completion.choices[0].message.content.strip()

try:
    # Überprüfen, ob das Modell sauberes JSON geliefert hat
    neue_konversation = json.loads(antwort_text)
    
    # System-Nachricht für die Identität deines Modells
    system_nachricht = {"role": "system", "content": "Du bist Gleneration, eine super freundliche und hilfsbereite Chat-KI! Du liebst es, mit Menschen zu schreiben und nutzt gerne viele Emojis! ✨"}
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
    neu_dataset = Dataset.from_dict({"messages": bestehende_liste})
    neu_dataset.push_to_hub(DATASET_REPO, token=HF_TOKEN)
    
    print("Erfolgreich abwechslungsreiche Emoji-Chatdaten auf Hugging Face gespeichert! 🎉")

except Exception as e:
    print(f"Fehler beim Verarbeiten des JSON: {e}")
    print(f"Rohtext der KI war: {antwort_text}")
