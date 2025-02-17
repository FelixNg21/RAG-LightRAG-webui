import os
from lightrag.lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
import textract
from typing import Literal

llm_model_kwargs = {"host": "http://localhost:11434", "options": {"num_ctx": 32768}}
class LightRagWrapper:
    def __init__(self, working_dir, llm_model_name, doc_dir, llm_model_kwargs=llm_model_kwargs, llm_model_max_async=4,
                 llm_model_max_token_size=32768, embedding_dim=768, max_token_size=8192):

        if not os.path.exists(working_dir):
            os.mkdir(working_dir)

        if llm_model_kwargs is None:
            llm_model_kwargs = llm_model_kwargs

        self.rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=ollama_model_complete,
            llm_model_name=llm_model_name,
            llm_model_max_async=llm_model_max_async,
            llm_model_max_token_size=llm_model_max_token_size,
            llm_model_kwargs=llm_model_kwargs,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=max_token_size,
                func=lambda texts: ollama_embed(
                    texts, embed_model="nomic-embed-text", host="http://localhost:11434"
                ),
            ),
        )
        self.doc_dir = doc_dir

    def ingest(self, path):
        text_content = textract.process(self.doc_dir+"/"+path)
        self.rag.insert(text_content.decode("utf-8"))

    def query(self, query_text, history: list = None, mode: Literal["local", "global", "hybrid", "naive","mix"]='hybrid'):
        return self.rag.query(query_text, param=QueryParam(mode=mode, conversation_history=history))

    def delete_by_doc_id(self, doc_id):
        self.rag.delete_by_doc_id(doc_id)

    def delete_by_entity_id(self, entity_id):
        self.rag.delete_by_entity(entity_id)
