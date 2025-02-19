import ollama
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

---
Answer the question based on the above context: {question}
"""


def extract_model_names(json):
    return [model['model'] for model in json['models']]


class OllamaInterface:
    def __init__(self, model: str, db: Chroma, collection_name="documents"):

        # self.ollama = ollama.Client("http://ollama:11434")
        self.ollama = ollama
        self.ollama_model_str = model
        self.chroma_path = CHROMA_PATH
        self.collection_name = collection_name
        self.db = db
        # load LLM into memory
        self.ollama.generate(model=self.ollama_model_str,
                             keep_alive=-1)

    def query(self, prompt: str, use_context: bool = True, history: list = None, history_limit: int = 5, context: list = None):

        try:
            if use_context:
                context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in context])
                prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                prompt_updated = prompt_template.format(context=context_text, question=prompt)
            else:
                prompt_updated = prompt
            if history:
                chat_history = history + [{"role": "user", "content": prompt_updated}]
            else:
                chat_history = [{"role": "user", "content": prompt_updated}]
            chat_history = chat_history[-(history_limit * 2):]
            return self.ollama.chat(
                model=self.ollama_model_str,
                messages=chat_history,
                stream=False,
                keep_alive=-1
            )
        except Exception as e:
            print(f"Error querying Ollama: {e}")
            return {"message": {"content": "An error occurred. Please try again."}}

    def get_context(self, prompt: str):
        return self.db.similarity_search_with_score(prompt)


    def get_details(self):
        details = self.ollama.list()
        return extract_model_names(details)

    def pull_model(self, model_name: str):
        try:
            self.ollama.pull(model_name)
        except:
            return "Model not found"

    def get_current_model(self):
        return extract_model_names(self.ollama.ps())

    def switch_model(self, model_name: str):
        self.ollama.generate(model=self.ollama_model_str,
                             keep_alive=0)
        self.ollama_model_str = model_name
        self.ollama.generate(model=self.ollama_model_str,
                             keep_alive=-1)
