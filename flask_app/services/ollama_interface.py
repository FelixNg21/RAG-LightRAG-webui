import ollama
from .get_embedding_func import get_embedding_function
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

---
Answer the question based on the above context: {question}
"""


class OllamaInterface:
    def __init__(self, model: str):

        self.ollama = ollama
        self.ollama_model_str = model
        self.chroma_path = CHROMA_PATH
        self.collection_name = "documents"
        self.db = Chroma(persist_directory=self.chroma_path,
                         embedding_function=get_embedding_function(),
                         collection_name=self.collection_name)

    def query_ollama(self, prompt: str):
        results = self.db.similarity_search_with_score(prompt, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt_updated = prompt_template.format(context=context_text, question=prompt)
        return self.ollama.chat(
            model=self.ollama_model_str,
            messages=[
                {
                    "role": "user",
                    "content": prompt_updated,
                }
            ],
            stream=False,
        )

    def get_db(self):
        return self.db

    def restart_db(self):
        self.db = Chroma(persist_directory=self.chroma_path,
                         embedding_function=get_embedding_function(),
                         collection_name=self.collection_name)

    def get_collection_name(self):
        return self.collection_name

    def get_details(self):
        return self.ollama.ps()


