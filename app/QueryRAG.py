# migrate functions from notebook to script

from get_embedding_func import get_embedding_function
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

---
Answer the question based on the above context: {question}
"""

class Query():
    def __init__(self):
        self.chroma_path = "chroma"
        self.db = Chroma(persist_directory=self.chroma_path, embedding_function=get_embedding_function())

    def query_rag(self, query_text: str):
        results = self.db.similarity_search_with_score(query_text, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = Ollama(model='mistral')
        response_text = model.invoke(prompt)

        # sources = [doc.metadata.get("id", None) for doc, _score in results]
        return f"Response: {response_text}"
