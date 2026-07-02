import os
from huggingface_hub import login
from transformers import LlamaConfig, LlamaForCausalLM

# 1. Login bei Hugging Face via GitHub Secret
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN fehlt!")

login(token=HF_TOKEN)

# Das Ziel-Repository für dein Modell
MODEL_REPO_ID = "GlaggleWeb/gleneration-v1"

print("🏗️ Erstelle den Bauplan (config.json) für Gleneration-v1...")

# Wir definieren eine feine, kompakte Architektur für den Start
# Das nutzt zwar das LlamaConfig-Format von HF, baut aber ein komplett leeres, 
# eigenes Modell von Grund auf ohne fremde Gewichte.
config = LlamaConfig(
    vocab_size=16000,             # Exakt abgestimmt auf deinen Tokenizer!
    hidden_size=512,              # Die Breite des Netzwerks
    intermediate_size=1376,       # Größe der inneren Schichten
    num_hidden_layers=8,          # 8 Schichten tief (perfekt für schnelles Training)
    num_attention_heads=8,        # Attention Heads
    max_position_embeddings=512,   # Maximale Chat-Länge (Kontext)
    pad_token_id=1,               # Entspricht [PAD] aus deinem Tokenizer
    bos_token_id=2,               # Entspricht [CLS]
    eos_token_id=3,               # Entspricht [SEP]
)

print("🧠 Initialisiere das leere Modell im RAM...")
# Erstellt die Gewichte (model.safetensors) – aktuell noch mit Zufallszahlen gefüllt
model = LlamaForCausalLM(config)

print("📤 Lade Modell-Bauplan und Gewichte zu Hugging Face hoch...")
# Schiebt config.json und model.safetensors direkt in dein Modell-Repo
model.push_to_hub(MODEL_REPO_ID, private=False)

print("🎉 FERTIG! Dein leeres Modell wurde erfolgreich initialisiert.")
