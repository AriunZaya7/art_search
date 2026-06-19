import chromadb
import os

client = chromadb.PersistentClient(path="chroma_db")
col = client.get_collection("artworks")
all_data = col.get(include=["metadatas"])
from collections import Counter
styles = [m.get("style", "unknown") for m in all_data["metadatas"]]
print(Counter(styles))
print(f"check script using: {os.path.abspath('chroma_db')}")