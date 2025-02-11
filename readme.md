# Inspiration
This project is my exploration into the world of RAGSs and how they can be used in real-world applications. I was coincidentally 
approached by a friend who wanted a tool to help him with his day-to-day work and I thought that this was a great opportunity
to build something that could be useful to him and also help me learn new things.

# Details
I attempted to build a frontend for that RAG that allows for the uploading of PDFs, the subsequent extraction of text from
those PDFs, the generation of embeddings for that text, and the ability to ask questions about the text. When I was approached
by my friend and learned about the outcome he wanted, simple RAG would not be sufficient. NaiveRAG as it is called, is naive 
in the sense that there is no knowledge base. The context that the answer is based on depends on how well the embeddings 
were generated. This is not ideal for my friend's use case as he wants to ask deeper questions than what is possible with RAG.

In my research I have found that RAG can be used with a knowledge base to provide more context to the answers. Microsoft 
has an implementation called GraphRAG. I found that the implementation was computationally complex and would not be 
feasible for me to test on my local machine. 

I then found a paper that provided a simpler solution to the problem, LightRAG. As the name suggests, LightRAG is an 
implementation that is "light" on compute, requiring a fraction of the API calls that GraphRAG requires. I have implemented
LightRAG in this project and is currently being tested.



# Technologies used
The LangChain library provides a bulk of the tools used in this project. It has tools for extracting text from PDFs, splitting
the text into chunks for embedding, as well as tools for interacting with the ChromaDB vector store.

The LLMs used in this project are from Ollama, which allows for me to run any open source LLM on my local machine.

The frontend is built with Flask and is currently very basic (ugly). The frontend allows a user to upload/delete PDFs, create
an embedding for them, clear the ChromaDB, and ask questions about the PDFs.

I've put off learning Javascript for a long time, and I am continuing to do so with this project. Lots of the logic is
provided by htmx and it's actually quite nice and simple to use.

The project also has a Gradio interface, that provides a much cleaner and less ugly UI for the user to interact with the RAG.

These technologies are all new to me, and I'm excited to see how they can be used in future projects.


# How to run

1. Clone the repo
2. Install the requirements
3. Run the app.py file in the `flask_app` directory.