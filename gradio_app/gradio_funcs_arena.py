import sys
import os
# # Get the directory containing the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
# # Get the parent directory
# parent_dir = os.path.dirname(current_dir)
# # Add the parent directory to sys.path
# sys.path.append(parent_dir)

import shutil
import re
import gradio as gr
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from services.chatlog_arena import ChatHistory, ChatMessage, db
from services.utils import generate_session_id
from services.lightrag_wrapper import LightRagWrapper
from services.ollama_interface import OllamaInterface
from services.chroma_db import Database
from services.document_loader import DocumentLoader

lightrag = LightRagWrapper(working_dir="./lightrag_docs", llm_model_name="deepseek-r1:8b",
                           doc_dir="/data/pdfs")
chroma_db = Database(chroma_path="chroma", collection_name="documents")
ollama = OllamaInterface("deepseek-r1:8b", chroma_db.db)
document_loader = DocumentLoader(chroma_db.db, collection_name="documents", data_path="data/pdfs")


engine = create_engine('sqlite:///chat_log_arena.db')
if not os.path.exists('chat_log_arena.db'):
    db.Model.metadata.create_all(engine)
SessionFactory=sessionmaker(bind=engine)
Session=scoped_session(SessionFactory)


SAVE_DIR = "./data/pdfs"
os.makedirs(SAVE_DIR, exist_ok=True)

# File Management
def save_files(files):
    """
    Save files to the data directory
    :param files: List of files
    :return: Message with the files uploaded
    """
    if not files:
        return "No files uploaded"
    saved_files = []
    for file in files:
        file_path = os.path.join(SAVE_DIR, os.path.basename(file.name))
        shutil.move(file.name, file_path)
        saved_files.append(file.name)
    return "Files uploaded: " + str(saved_files)


def update_files():
    """
    Update the list of files in the directory
    :return: CheckboxGroup with the updated files
    """
    updated_files = list_files()
    return gr.CheckboxGroup(label="Uploaded Files", choices=updated_files)


def list_files():
    """
    List the files in the data directory
    :return: List of files or message if no files
    """
    files = os.listdir(SAVE_DIR)
    return files if files else "No files uploaded"


def process_files(selected_files):
    """
    Process the selected files for both NaiveRAG and LightRAG
    :param selected_files: List of selected files
    :return: Message with the files processed
    """
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
def user(user_message, history: list, session_id=None):
    """
    Handle user message
    :param user_message: User message
    :param history: Chat history
    :return: String for textbox, Updated history, user message, RAG type
    """
    history = history or []
    if not history and session_id is None:
        session_id = generate_session_id()
    history = history + [{"role": "user", "content": user_message}]
    return "", history, user_message, session_id


def get_context(history, user_message):
    """
    Get context based on the user message
    :param history: Chat history
    :param user_message: User message
    """
    context = ollama.get_context(user_message)
    return context, history, user_message


def handle_reasoning(model_response):
    """
    Handle reasoning in the model response
    :param model_response: Model response
    :return: Model response, Thinking
    """
    if "<think>" in model_response:
        thinking = re.findall(r"<think>.*?</think>", model_response, flags=re.DOTALL)
        model_response = re.sub(r"<think>.*?</think>", "", model_response, flags=re.DOTALL)
        return model_response, thinking
    return model_response, None


def assistant(history: list, user_message, rag_type, context, session_id=None):
    """
    Handle assistant response
    :param history: Chat history
    :param user_message: User message
    :param rag_type: RAG type
    :param context: Context from vector search
    """
    if rag_type == "LightRAG":
        content = lightrag.query(user_message, history=history)
    else:
        response = ollama.query(user_message, use_context=True, history=history, context=context)
        content = response["message"]["content"]
    content, thinking = handle_reasoning(content)

    history.append({"role": "assistant", "content": ""})

    for character in content:
        history[-1]["content"] += character
        yield history

    save_chat_history(history, rag_type, session_id)
    return history, session_id


# Chat History Management
def save_chat_history(history, rag_type, session_id=None):
    """
    Save chat history to the database
    :param history: Chat history
    :param rag_type: RAG type
    """
    session = Session()

    try:
        if session_id is None:
            session_id = generate_session_id()
        chat = session.query(ChatHistory).filter_by(session_id=session_id).first()

        if not chat:
            chat = ChatHistory(session_id=session_id)
            session.add(chat)
            session.flush()

        session.query(ChatMessage).filter_by(
            chat_id=chat.id,
            rag_type=rag_type
        ).delete()
        for message in history:
            msg = ChatMessage(
                chat_id = chat.id,
                rag_type = rag_type,
                role = message["role"],
                content = message["content"]
            )
            session.add(msg)
        session.commit()
    finally:
        Session.remove()
    return gr.update(choices=get_chat_histories())

def get_chat_histories():
    """
    Get chat histories from the database
    :return: Choices for dropdown, Initial value for dropdown
    """
    session = Session()
    try:
        histories = session.query(ChatHistory).order_by(ChatHistory.timestamp.desc()).all()
        choices = [history.session_id for history in histories]
        initial_value = choices[0] if choices else None

    finally:
        Session.remove()
    return choices, initial_value

def load_chat_history(session_data):
    """
    Load chat history from the database
    :param session_data: Session data
    :return: Chat history
    """
    session = Session()
    if not session_data:
        return [], []

    session_id = session_data.get('value') if isinstance(session_data, dict) else session_data
    try:
        chats = session.query(ChatHistory).filter_by(session_id=session_id).all()
        if not chats:
            return [], []
        print("Chat", chats)
        naive_chat = [{"role": message.role, "content": message.content} for chat in chats for message in chat.naive_messages]
        lightrag_chat = [{"role": message.role, "content": message.content} for chat in chats for message in chat.light_messages]


        return naive_chat, lightrag_chat
    finally:
        Session.remove()

def refresh_histories():
    """
    Refresh chat histories
    :return: Updated choices and value
    """
    choices, value = get_chat_histories()
    return gr.update(choices=choices, value=value)

def delete_chat(session_data):
    """
    Delete chat history from the database
    """
    session = Session()
    if not session_data:
        return gr.update(), gr.update()
    session_id = session_data.get('value') if isinstance(session_data, dict) else session_data
    try:
        chat = session.query(ChatHistory).filter_by(session_id=session_id).first()
        if chat:
            session.delete(chat)
            session.commit()
    finally:
        Session.remove()
    choices, value = get_chat_histories()
    return [], [], gr.update(choices=choices, value=value)

def pdf_viewer(history):
    if history[-1]["role"] == "assistant":
        return True
    return False