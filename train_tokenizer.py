import os
import json
import requests
from huggingface_hub import HfApi, login
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 1. Login bei Hugging Face via GitHub Secret
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN fehlt in den Umgebungsvariablen!")

login(token=HF_TOKEN)
REPO_ID = "GlaggleWeb/glenerationwissen"
api = HfApi()

# 2. wissen.json direkt in den RAM laden
print("🌐 Streame wissen.json direkt aus der Hugging Face Cloud...")
file_url = f"https://huggingface.co/datasets/{REPO_ID}/raw/main/wissen.json"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

response = requests.get(file_url, headers=headers)
if response.status_code != 200:
    raise Exception(f"Fehler beim Laden von Hugging Face: {response.status_code}")

gesamtes_wissen = response.json()

# 3. Texte im Arbeitsspeicher extrahieren
trainings_texte = []
for chat in gesamte_wissen:
    for nachricht in chat:
        if nachricht["role"] in ["user", "assistant"]:
            trainings_texte.append(nachricht["content"])

print(f"🧠 {len(trainings_texte)} Nachrichten werden jetzt analysiert...")

# 4. BPE-Tokenizer konfigurieren & trainieren
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=16000, 
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
)

print("⚙️ Trainiere Tokenizer (Vokabular-Erstellung)...")
tokenizer.train_from_iterator(trainings_texte, trainer)

# 5. Direkt aus dem RAM wieder hochladen
print("📤 Lade fertige tokenizer.json zu Hugging Face hoch...")
tokenizer_json_string = tokenizer.to_str()
tokenizer_bytes = tokenizer_json_string.encode("utf-8")

api.upload_file(
    path_or_fileobj=tokenizer_bytes,
    path_in_repo="tokenizer.json",
    repo_id=REPO_ID,
    repo_type="dataset"
)

print("🎉 FERTIG! Die tokenizer.json liegt jetzt in deinem HF-Dataset.")
