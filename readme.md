# Inspiration
This is a project that pushes me to experiment with new technologies and learn new things. I've taken an interest to RAGs
recently and wanted to see what application I can build with them. This project is a simple example of a RAG agent that 
can be hosted locally and can answer questions about the content of your PDFs.

# Technologies used
Langchain is used to extract the text from the PDFs. The text is then split into chunks that can be added to a ChromaDB 
vector store.

Embeddings must then be generated for the text chunks. This is done using the Ollama embeddings. Currently, the 
'nomic-embed-text' is the model used for the embeddings.

The LLM behind the RAG is also from Ollama. I'm currently using the mistral model as that was the newest LLM made available
via Ollama.

The frontend is built with Flask and is currently very basic (ugly). The frontend allows a user to upload/delete PDFs, create
an embedding for them, clear the ChromaDB, and ask questions about the PDFs.

I've put off learning Javascript for a long time, and I am continuing to do so with this project. Lots of the logic is
provided by htmx and it's actually quite nice and simple to use.

These technologies are all new to me and I'm excited to see how they can be used in future projects.


# How to run

1. Clone the repo
2. Install the requirements
3. Run the app/run.py file