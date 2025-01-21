from langchain_community.embeddings.ollama import OllamaEmbeddings


def get_embedding_function(embedding_model="nomic-embed-text"):
    embeddings = OllamaEmbeddings(
        model=embedding_model,
    )
    return embeddings
