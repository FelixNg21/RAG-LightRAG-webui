from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from services.chroma_db import Database


class DocumentLoader:
    """
    DocumentLoader class to load and split documents for use in RAG application
    """

    def __init__(self, db: Database, collection_name="documents", data_path="data/pdfs"):
        self.data_path = data_path
        self.loader = PyPDFDirectoryLoader(self.data_path)
        self.db = db
        self.collection_name = collection_name

    def load_documents(self, file_paths=None):
        if file_paths:
            documents = []
            for file_path in file_paths:
                full_path = f"{file_path}"
                loader = PyPDFLoader(full_path)
                documents.extend(loader.load())
            return documents
        else:
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
        chunks_with_ids = self.calculate_chunk_ids(chunks)

        existing_items = self.db.get()
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

    def delete_document(self, docs_to_delete):
        docs_to_delete = [docs_to_delete]
        all_docs = self.get_documents()["ids"]
        target_docs = [target for target in all_docs for doc in docs_to_delete if doc in target]
        if target_docs:
            self.db.delete(target_docs)

    def get_documents(self):
        collection = self.db.get()
        return collection

    def ingest(self, file_path):
        documents = self.load_documents(file_path)
        chunks = self.split_documents(documents)
        self.add_to_chroma(chunks)