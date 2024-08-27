# migrate functions from notebook to script

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from chromadb import PersistentClient
import shutil
from app.services.get_embedding_func import get_embedding_function


class DocumentLoader:
    """
    DocumentLoader class to load and split documents for use in RAG application
    """

    def __init__(self):
        self.data_path = "data/pdfs"
        self.chrome_path = "chroma"
        self.collection_name = "documents"
        self.loader = PyPDFDirectoryLoader(self.data_path)
        self.db = None

    def load_documents(self):
        return self.loader.load()

    def split_documents(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

    def add_to_chroma(self, chunks: list[Document]):
        self.db = Chroma(
            persist_directory=self.chrome_path,
            embedding_function=get_embedding_function(),
            collection_name=self.collection_name,
        )
        chunks_with_ids = self.calculate_chunk_ids(chunks)

        existing_items = self.db.get(include=[])
        existing_ids = set(existing_items["ids"])
        print(f'Existing items: {len(existing_ids)}')

        if new_chunks := [
            chunk
            for chunk in chunks_with_ids
            if chunk.metadata["id"] not in existing_ids
        ]:
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            self.db.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            print("No new chunks to add")
        print(f'New items: {len(new_chunks)}')

    def calculate_chunk_ids(self, chunks: list[Document]):
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            chunk.metadata["id"] = chunk_id
        return chunks

    def clear_database(self):
        try:
            chroma_client = PersistentClient(self.chrome_path)
            chroma_client.delete_collection(self.collection_name)
            shutil.rmtree(self.chrome_path)
        except Exception as e:
            raise e
