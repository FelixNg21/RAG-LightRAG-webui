from flask import request, jsonify
from app.app import app
from app.services.document_loader import DocumentLoader

document_loader = DocumentLoader()


@app.route("/upload", methods=["POST"])  #TODO
def upload_and_store():
    # takes in pdf files and stores them in the data/pdfs directory
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith(".pdf"):
        file.save(f"data/pdfs/{file.filename}")
        return jsonify({"message": f"Document uploaded successfully: {file.filename}"}), 200
    else:
        return jsonify({"error": "Invalid file type, must be pdf"}), 400


@app.route("/vectorize", methods=["POST"])  #TODO
def vectorize():
    # takes in a list of pdf files and vectorizes them
    documents = document_loader.load_documents()
    chunks = document_loader.split_documents(documents)
    document_loader.add_to_chroma(chunks)
    return jsonify({"message": "Documents vectorized successfully"}), 200


@app.route("/query", methods=["GET"])  #TODO
def query():
    # takes in a query and returns the LLM answer
    pass
