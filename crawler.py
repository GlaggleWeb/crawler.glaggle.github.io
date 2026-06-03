import os
import requests
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)

def crawl_and_index(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find('title').text if soup.find('title') else url
            text_content = soup.get_text(separator=' ', strip=True)[:3000]
            
            point_id = abs(hash(url)) % 10000000
            
            # WICHTIG: Wir nutzen client.upsert und übergeben den Text im 'vectors'-Feld
            # Damit weiß Qdrant Cloud sofort, dass sie den Vektor berechnen soll
            client.upsert(
                collection_name="web_pages",
                points=[
                    PointStruct(
                        id=point_id,
                        # Wir übergeben den Text direkt an das Cloud-Modell
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
