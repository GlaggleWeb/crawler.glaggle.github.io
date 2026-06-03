import os
import requests
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
# NEU: Das richtige Format-Objekt von Qdrant importieren
from qdrant_client.models import PointStruct 

# Zugangsdaten aus den GitHub Secrets laden
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

# check_compatibility=False hinzugefügt, um die Warnung zu unterdrücken
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)

def crawl_and_index(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find('title').text if soup.find('title') else url
            
            # Text extrahieren (ersten 3000 Zeichen)
            text_content = soup.get_text(separator=' ', strip=True)[:3000]
            
            # Generiere eine saubere ID (Muss eine Zahl zwischen 0 und 9223372036854775807 sein)
            point_id = abs(hash(url)) % 10000000
            
            # Daten an Qdrant senden (Jetzt im richtigen PointStruct-Format!)
            client.upload_points(
                collection_name="web_pages",
                points=[
                    PointStruct(
                        id=point_id,
                        vector={},  # Qdrant übernimmt das Embedding in der Cloud
                        payload={
                            "url": url,
                            "title": title,
                            "text": text_content
                        }
                    )
                ]
            )
            print(f"Erfolgreich indexiert: {title}")
    except Exception as e:
        print(f"Fehler beim Crawlen von {url}: {e}")

# Test-URLs
urls_to_crawl = [
    "https://de.wikipedia.org/wiki/Affen",
    "https://www.zoo.ch"
]

for url in urls_to_crawl:
    crawl_and_index(url)
