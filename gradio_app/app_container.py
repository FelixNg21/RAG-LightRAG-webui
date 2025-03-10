import os
from services.chroma_db import Database
from services.document_loader import DocumentLoader
from services.lightrag_wrapper import LightRagWrapper
from gradio_funcs import HistoryManager, FileManager, ChatManager


class ApplicationContainer:
    """Centralized dependency management"""

    def __init__(self):
        self.services = {}
        self.load_config()
        self.initialize_services()

    def load_config(self):
        self.config = {
            "CHROMA_PATH": os.environ.get("CHROMA_PATH"),
            "CHROMA_COLLECTION": os.environ.get("CHROMA_COLLECTION"),
            "SAVE_DIR": os.environ.get("DOCUMENT_DIR"),
            "LIGHTRAG_DIR": os.environ.get("LIGHTRAG_DIR"),
            "LR_INGEST": os.environ.get("LR_INGEST"),
            "LR_GENERATE": os.environ.get("LR_GENERATE"),
            "CHAT_LOG_DB": os.environ.get("CHAT_LOG_DB")
        }

    def initialize_services(self):
        # Core infrastructure
        chroma_db = Database(chroma_path=self.config["CHROMA_PATH"],
                            collection_name=self.config["CHROMA_COLLECTION"])
        self.services["chroma_db"] = chroma_db

        # Service layer
        document_loader = DocumentLoader(db=chroma_db.db,
                                       collection_name=self.config["CHROMA_COLLECTION"],
                                       data_path=self.config["SAVE_DIR"])
        self.services["document_loader"] = document_loader

        lightrag = LightRagWrapper(working_dir=self.config["LIGHTRAG_DIR"],
                                  llm_model_ingest=self.config["LR_INGEST"],
                                  llm_model_gen=self.config["LR_GENERATE"],
                                  doc_dir=self.config["SAVE_DIR"])
        self.services["lightrag"] = lightrag

        # Manager classes
        history_manager = HistoryManager(db_path=self.config["CHAT_LOG_DB"])
        self.services["history_manager"] = history_manager

        file_manager = FileManager(save_dir=self.config["SAVE_DIR"],
                                  document_loader=document_loader,
                                  lightrag=lightrag)
        self.services["file_manager"] = file_manager

        chat_manager = ChatManager(chroma_db=chroma_db,
                                 lightrag_instance=lightrag,
                                 history_manager=history_manager)
        self.services["chat_manager"] = chat_manager

    def get(self, service_name):
        return self.services.get(service_name)