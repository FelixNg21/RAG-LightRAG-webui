from langchain_chroma import Chroma
from chromadb import PersistentClient
from .get_embedding_func import get_embedding_function
class Database:
    def __init__(self, chroma_path="chroma", collection_name="documents"):
        self.collection_name = collection_name
        self.chroma_path = chroma_path
        self.db = Chroma(persist_directory=chroma_path,
                         embedding_function=get_embedding_function(),
                         collection_name=self.collection_name)
        self.chroma_client = PersistentClient(self.chroma_path)

    def clear_database(self):
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception as e:
            print(f"Error clearing database: {e}")

    def restart_database(self):
        try:
            self.clear_database()
        except:
            print("Error clearing database")
        self.db = Chroma(persist_directory=self.chroma_path,
                         embedding_function=get_embedding_function(),
                         collection_name=self.collection_name)

    def similarity_search_with_score(self, query: str, k=5):
        return self.db.similarity_search_with_score(query, k=k)

    def get_collection_name(self):
        return self.collection_name