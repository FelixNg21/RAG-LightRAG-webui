from flask import request, jsonify, Blueprint, Response, render_template, url_for, redirect, make_response
from app.services.document_loader import DocumentLoader
from app.services.ollama_interface import OllamaInterface
import os

ollama_interface = OllamaInterface(model="mistral")
document_loader = DocumentLoader(ollama_interface.get_db())
route_api = Blueprint("route_api", __name__)


@route_api.route("/upload", methods=["POST"])  # TODO
def upload_and_store():
    if request.method != "POST":
        return "Method not allowed"
    # takes in pdf files and stores them in the data/pdfs directory
    if 'files' not in request.files:
        return "No file part"

    files = request.files.getlist("files")

    for file in files:
        if file.filename == '':
            return "No selected file"

        if file and file.filename.endswith(".pdf"):
            os.makedirs("data/pdfs", exist_ok=True)
            file.save(f"data/pdfs/{file.filename}")

        else:
            return "File must be a pdf"

    if len(files) == 1:
        response = make_response("File uploaded successfully", 200)
        response.headers['HX-Trigger'] = 'newFileUpload'
        return response
    response = make_response("File(s) uploaded successfully", 200)
    response.headers['HX-Trigger'] = 'newFileUpload'
    return response


@route_api.route("/listfiles", methods=["GET"])
def list_files():
    if request.method != "GET":
        return "Method not allowed"
    files = os.listdir("data/pdfs")
    response = ''
    if not files:
        return "No files uploaded"
    for file in files:
        response += f"<input type='checkbox' name='file' value='{file}' hx-trigger='true'> {file}<br>"
    return response


@route_api.route("/vectorize", methods=["POST"])  # TODO
def vectorize():
    # takes in a list of pdf files and vectorizes them
    if request.method != "POST":
        return "Method not allowed"
    print("db exists" if ollama_interface.get_db() else "db does not exist")
    try:
        documents = document_loader.load_documents()
        chunks = document_loader.split_documents(documents)
        document_loader.add_to_chroma(chunks, ollama_interface)
    except:
        reinitialize_db()

    return "Files vectorized successfully"


@route_api.route("/query", methods=["POST"])  # TODO
async def query():
    # takes in a query and returns a response
    if request.method != "POST":
        return "Method not allowed"
    query_text = request.form.get("query")

    result = ollama_interface.query_ollama(query_text)

    def generate():
        for chunk in result:
            yield chunk['message']['content']

    return Response(generate(), content_type="text/plain")


@route_api.route("/delete", methods=["POST"])
def delete():
    # deletes all pdf files
    if request.method != "POST":
        return "Method not allowed"
    files = request.form.getlist("file")
    for file in files:
        try:
            os.remove(f"data/pdfs/{file}")
        except FileNotFoundError:
            return f"File {file} not found"
    if len(files) == 0:
        return "No files selected"
    response = make_response("Selected files deleted", 200)
    response.headers['HX-Trigger'] = 'fileDeleted'
    return response

@route_api.route("/reinitialize-db", methods=["POST"])
def reinitialize_db():
    if request.method != "POST":
        return "Method not allowed"
    ollama_interface.restart_db()
    return "Database reinitialized"

@route_api.route("/clear-db", methods=["POST"])
def clear_db():
    if request.method != "POST":
        return "Method not allowed"
    document_loader.clear_database()
