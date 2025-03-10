from langchain_chroma import Chroma
from chromadb import PersistentClient
from .get_embedding_func import get_embedding_function

class Database:
    def __init__(self, chroma_path="chroma", collection_name="documents"):
        self.collection_name = collection_name
        self.chroma_path = chroma_path
        self.persistent_client = PersistentClient(self.chroma_path)
        self.db = Chroma(client=self.persistent_client,
                         embedding_function=get_embedding_function(),
                         collection_name=self.collection_name)

    def clear_database(self):
        try:
            self.db.delete_collection()
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

    def delete(self, doc_id):
        self.db.delete(doc_id)

    def get(self):
        return self.db.get(include=None)

    def add_documents(self, new_chunks, ids):
        self.db.add_documents(new_chunks, ids=ids)