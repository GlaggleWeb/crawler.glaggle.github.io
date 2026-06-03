import os
import requests
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)

COLLECTION_NAME = "web_pages"

# NEU: Automatische Erstellung der Collection, falls sie fehlt
try:
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' existiert nicht. Wird erstellt...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                # Wir nutzen das standardmäßige, integrierte Text-Modell von Qdrant
                "text": VectorParams(size=384, distance=Distance.COSINE)
            }
        )
        print(f"Collection '{COLLECTION_NAME}' erfolgreich erstellt!")
except Exception as e:
    print(f"Hinweis beim Überprüfen der Collection: {e}")


def crawl_and_index(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find('title').text if soup.find('title') else url
            text_content = soup.get_text(separator=' ', strip=True)[:3000]
            
            point_id = abs(hash(url)) % 10000000
            
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=point_id,
                        vector={"text": text_content}, 
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

urls_to_crawl = [
    "https://de.wikipedia.org/wiki/Affen",
    "https://www.zoo.ch"
]

for url in urls_to_crawl:
    crawl_and_index(url)
