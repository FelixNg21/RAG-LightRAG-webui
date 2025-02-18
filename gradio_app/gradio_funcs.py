import shutil
import re
import gradio as gr

from services.chatlog import ChatHistory, ChatMessage, db
from services.utils import generate_session_id
from flask import Flask
import os


from services.lightrag_wrapper import LightRagWrapper
from services.ollama_interface import OllamaInterface
from services.chroma_db import Database
from services.document_loader import DocumentLoader

lightrag = LightRagWrapper(working_dir="lightrag_docs", llm_model_name="deepseek-r1:8b",
                           doc_dir="./data/pdfs")
chroma_db = Database(chroma_path="chroma", collection_name="documents")
ollama = OllamaInterface("deepseek-r1:8b", chroma_db.db)
document_loader = DocumentLoader(chroma_db.db, collection_name="documents", data_path="data/pdfs")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat_log.db'
db.init_app(app)

with app.app_context():
    db.create_all()

SAVE_DIR = "data/pdfs"
os.makedirs(SAVE_DIR, exist_ok=True)

# File Management
def save_files(files):
    if not files:
        return "No files uploaded"
    saved_files = []
    for file in files:
        file_path = os.path.join(SAVE_DIR, os.path.basename(file.name))
        shutil.move(file.name, file_path)
        saved_files.append(file.name)
    return "Files uploaded: " + str(saved_files)


def update_files():
    updated_files = list_files()
    return gr.CheckboxGroup(label="Uploaded Files", choices=updated_files)


def list_files():
    files = os.listdir(SAVE_DIR)
    return files if files else "No files uploaded"


def process_files(selected_files):
    if not selected_files:
        return "No files selected"
    try:
        # Processing for NaiveRAG
        documents = document_loader.load_documents(selected_files)
        chunks = document_loader.split_documents(documents)
        document_loader.add_to_chroma(chunks)

        # Processing for LightRAG
        for file in selected_files:
            print('Ingesting file:', file)
            lightrag.ingest(file)

    except Exception as e:
        return "Error processing files: " + str(e)

    return "Files processed selected files: " + str(selected_files)


# Chat functions
def user(user_message, history: list, rag_type):
    history = history or []
    history = history + [{"role": "user", "content": user_message}]
    return "", history, user_message, rag_type


def get_context(history, user_message, rag_type):
    context = ollama.get_context(user_message)
    return context, history, user_message, rag_type


def handle_reasoning(model_response):
    if "<think>" in model_response:
        thinking = re.findall(r"<think>.*?</think>", model_response, flags=re.DOTALL)
        model_response = re.sub(r"<think>.*?</think>", "", model_response, flags=re.DOTALL)
        return model_response, thinking
    return model_response, None


def assistant(history: list, user_message, rag_type, context):
    if rag_type == "LightRAG":
        content = lightrag.query(user_message, history=history)
    else:
        response = ollama.query(user_message, use_context=True, history=history, context=context)
        content = response["message"]["content"]
    content, thinking = handle_reasoning(content)

    history.append({"role": "assistant", "content": ""})


    streamed_response = ""
    for character in content:
        streamed_response += character
        history[-1]["content"] += character
        yield history

    save_chat_history(history + [{"role": "assistant", "content": content}])

def save_chat_history(history):
    with app.app_context():
        chat = ChatHistory(session_id=generate_session_id())
        db.session.add(chat)
        db.session.flush()
        for message in history:
            msg = ChatMessage(
                chat_id = chat.id,
                role = message["role"],
                content = message["content"]
            )
            db.session.add(msg)
        db.session.commit()
    return gr.update(choices=get_chat_histories())

def get_chat_histories():
    with app.app_context():
        histories = ChatHistory.query.order_by(ChatHistory.timestamp.desc()).all()
        choices = [{
            'value': history.session_id
        } for history in histories]
        initial_value = choices[0] if choices else None
        return choices, initial_value

def load_chat_history(session_data):
    if not session_data:
        return []

    session_id = session_data.get('value') if isinstance(session_data, dict) else session_data
    with app.app_context():
        chat = ChatHistory.query.filter_by(session_id=session_id).first()
        if not chat:
            return []
        messages = chat.messages
        return [{"role": message.role, "content": message.content} for message in messages]

def refresh_histories():
    choices, value = get_chat_histories()
    return gr.update(choices=choices, value=value, interactive=True)

def delete_chat(session_data):
    if not session_data:
        return gr.update(), gr.update()
    session_id = session_data.get('value') if isinstance(session_data, dict) else session_data
    with app.app_context():
        chat = ChatHistory.query.filter_by(session_id=session_id).first()
        if chat:
            db.session.delete(chat)
            db.session.commit()
    choices, value = get_chat_histories()
    return [], gr.update(choices=choices, value=value)

def pdf_viewer(history):
    if history[-1]["role"] == "assistant":
        return True
    return False