import asyncio

from flask import request, jsonify, Blueprint, Response, render_template, url_for, redirect
from app.services.document_loader import DocumentLoader
from app.services.ollama_interface import OllamaInterface
import os

document_loader = DocumentLoader()
ollama_interface = OllamaInterface(model="mistral")

route_api = Blueprint("route_api", __name__)


@route_api.route("/upload", methods=["POST"])  #TODO
def upload_and_store():
    if request.method != "POST":
        return render_template("index.html", message_upload="Method not allowed")
    # takes in pdf files and stores them in the data/pdfs directory
    if 'file' not in request.files:
        return render_template("index.html", message_upload="No file part")

    file = request.files['file']

    if file.filename == '':
        return render_template("index.html", message_upload="No selected file")

    if file and file.filename.endswith(".pdf"):
        os.makedirs("data/pdfs", exist_ok=True)
        file.save(f"data/pdfs/{file.filename}")
        return render_template("index.html", message_upload="File uploaded successfully", uploaded=True,
                               files=os.listdir("data/pdfs"))
    else:
        return render_template("index.html", message_upload="File must be a pdf")


@route_api.route("/vectorize", methods=["POST"])  #TODO
def vectorize():
    # takes in a list of pdf files and vectorizes them
    documents = document_loader.load_documents()
    chunks = document_loader.split_documents(documents)
    document_loader.add_to_chroma(chunks)
    return render_template("index.html", message_vectorize="Documents vectorized successfully",
                           files=os.listdir("data/pdfs"))


@route_api.route("/query", methods=["POST"])  #TODO
async def query():
    # takes in a query and returns a response
    query_text = request.form.get("query")

    result = ollama_interface.query_ollama(query_text)

    def generate():
        for chunk in result:
            yield chunk['message']['content']

    return Response(generate(), content_type="text/plain")


@route_api.route("/delete", methods=["POST"])
def delete():
    # deletes all pdf files
    for file in os.listdir("data/pdfs"):
        os.remove(f"data/pdfs/{file}")
    return render_template("index.html", message_delete="Files deleted successfully", files=os.listdir("data/pdfs"))
