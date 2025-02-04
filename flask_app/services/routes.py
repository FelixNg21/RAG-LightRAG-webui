from flask import request, Blueprint, make_response, jsonify, url_for
from .document_loader import DocumentLoader
from .ollama_interface import OllamaInterface
import os
from .chatlog import db, ChatLog
import uuid
from .chroma_db import Database

chroma_db = Database(chroma_path="chroma", collection_name="documents")
ollama_interface = OllamaInterface(model="deepseek-r1:14b", db=chroma_db.db)
document_loader = DocumentLoader(db=chroma_db.db, collection_name="documents")
route_api = Blueprint("route_api", __name__)

"""
Chat Management
"""


@route_api.route("/query", methods=["POST"])
async def query():
    if request.method == 'POST':
        query_text = request.form.get("query")
        session_id = request.cookies.get("sessionID")
        if query_text == '':
            print("No query text")

        response_text = chat(query_text)
        print(response_text)
        # Save chat to database
        chat_log = ChatLog(session_id=session_id, user_query=query_text, chatbot_response=response_text)
        db.session.add(chat_log)
        db.session.commit()

        response = ''
        response += div_generator("user-query", f'{query_text}')
        response += div_generator("chatbot-response", f'{response_text}')

        return response


def div_generator(classname, text):
    return f"<div class='{classname}'>{text}</div> "


def chat(query_text):
    result = ollama_interface.query_ollama(query_text)

    return result['message']['content']


@route_api.route('/load-chat', methods=["GET"])
def load_chat():
    session_id = request.args.get("session_id")
    chat_logs = ChatLog.query.filter_by(session_id=session_id).all()
    chat_data = [
        {"user_query": log.user_query, "chatbot_response": log.chatbot_response}
        for log in chat_logs
    ]
    return jsonify(chat_data)


@route_api.route('/view-chats', methods=["GET"])
def view_chats():
    chat_logs = ChatLog.query.all()
    chat_data = [
        {"session_id": log.session_id, "user_query": log.user_query, "chatbot_response": log.chatbot_response}
        for log in chat_logs
    ]
    return jsonify(chat_data)


#TODO: modify to clear chat since this function makes a new session
@route_api.route('/new-session', methods=["POST"])
def new_session():
    response = make_response('New Chat', 200)
    response.headers['HX-Trigger'] = 'newSession'
    return response

    # return jsonify({"message": "New session created", "session_id": new_session_id}), 200


def generate_session_id():
    return str(uuid.uuid4())


"""
PDF Management
"""


@route_api.route("/upload", methods=["POST"])
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


@route_api.route("/vectorize", methods=["POST"])
def vectorize():
    # takes in a list of pdf files and vectorizes them
    if request.method != "POST":
        return "Method not allowed"
    try:
        documents = document_loader.load_documents()
        chunks = document_loader.split_documents(documents)
        document_loader.add_to_chroma(chunks, ollama_interface)
    except:
        reinitialize_db()

    return "Files vectorized successfully"


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
    try:
        chroma_db.restart_database()
    except:
        return "Error reinitializing database"
    return "Database reinitialized"


@route_api.route("/clear-db", methods=["POST"])
def clear_db():
    if request.method != "POST":
        return "Method not allowed"
    try:
        chroma_db.clear_database()
    except:
        return "Error clearing database"

    return "Database cleared"


"""
Model Management
"""


@route_api.route('/model-details', methods=["GET"])
def model_details():
    if request.method != "GET":
        return "Method not allowed"
    details = ollama_interface.get_details()
    response = ''
    for detail in details:
        response += f"<option value='{detail}'>{detail}</option>"
    return response


@route_api.route("/pull-models", methods=["POST"])
def pull_model():
    if request.method != "POST":
        return "Method not allowed"
    model_name = request.form.get("model-name")
    if model_name == '':
        return "No model entered"
    ollama_interface.pull_model(model_name)
    return "Model pulled successfully"


@route_api.route('/current-model', methods=["GET"])
def current_model():
    if request.method != "GET":
        return "Method not allowed"
    current_models = ollama_interface.get_current_model()
    response = ''
    for model in current_models:
        response += f"<option value='{model}'>{model}</option>"
    return response


@route_api.route('/switch-model', methods=["POST"])
def switch_model():
    if request.method != "POST":
        return "Method not allowed"
    model_name = request.form.get("model-name")
    if model_name == '':
        return "No model entered"
    ollama_interface.switch_model(model_name)
    response = make_response("Model switched successfully", 200)
    response.headers['HX-Trigger'] = 'modelSwitched'
    return response
