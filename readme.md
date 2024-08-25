# Locally hosted RAG agent

This is a simple example of a RAG agent that is hosted locally. It provides a usable, but ugly, frontend where a user can
upload the PDFs that they would like the agent to know the context of. The agent will then be able to answer questions about
the content of the PDFs.

The webapp itself is served from /app/run.py and is built with Flask. The agent can be customized to use any model that is
available from Ollama. The embeddings are also from the Ollama embeddings resulting in a no cost solution.

Further work is needed to make the frontend visually appealing and provide more customization options in the frontend.