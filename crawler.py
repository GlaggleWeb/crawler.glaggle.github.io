import os
import requests
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient

# Zugangsdaten sicher aus den GitHub Secrets laden
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def crawl_and_index(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find('title').text if soup.find('title') else url
            
            # Text extrahieren (z.B. ersten 3000 Zeichen)
            text_content = soup.get_text(separator=' ', strip=True)[:3000]
            
            # Daten an Qdrant senden
            client.upload_points(
                collection_name="web_pages",
                points=[
                    {
                        "id": hash(url) % 10000000,  # Eine eindeutige ID generieren
                        "vector": {},  # Leer lassen, da Qdrant die Vektoren selbst generiert
                        "payload": {
                            "url": url,
                            "title": title,
                            "text": text_content
                        }
                    }
                ]
            )
            print(f"Erfolgreich indexiert: {title}")
    except Exception as e:
        print(f"Fehler beim Crawlen von {url}: {e}")

# Test-URLs (Hier kannst du später deine Logik einbauen, um Links automatisch zu finden)
urls_to_crawl = [
    "https://de.wikipedia.org/wiki/Affen",
    "https://www.zoo.ch"
]

for url in urls_to_crawl:
    crawl_and_index(url)
