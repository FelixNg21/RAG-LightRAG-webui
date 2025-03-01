import os
import shutil
import re
import gradio as gr
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from services.chatlog import ChatHistory, ChatMessage, ChatHistoryArena, ChatMessageArena, db
from services.utils import generate_session_id
from services.lightrag_wrapper import LightRagWrapper
from services.ollama_interface import OllamaInterface
from services.chroma_db import Database
from services.document_loader import DocumentLoader

lightrag = LightRagWrapper(working_dir="./lightrag_docs", llm_model_name="deepseek-r1:8b",
                           doc_dir="./data/pdfs")
chroma_db = Database(chroma_path="chroma", collection_name="documents")
ollama = OllamaInterface("deepseek-r1:8b", chroma_db.db)
document_loader = DocumentLoader(chroma_db.db, collection_name="documents", data_path="./data/pdfs")


engine = create_engine('sqlite:///chat_log.db')
if not os.path.exists('chat_log.db'):
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
        document_loader.ingest(selected_files)

        # Processing for LightRAG
        lightrag.ingest(selected_files)

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


def assistant(history: list, user_message, rag_type, context, session_id=None, arena_flag=False):
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

    if arena_flag:
        save_chat_history_arena(history, rag_type, session_id)
    else:
        save_chat_history(history, rag_type, session_id)
    return history, session_id


# Chat History Management
def save_chat_history(history, rag_type, session_id=None):
    """
    Save chat history to the database
    :param history: Chat history
    :rag_type: RAG type
    :session_id: Session ID
    """
    session = Session()
    try:
        if session_id is None:
            session_id = generate_session_id()
        chat = session.query(ChatHistory).filter_by(session_id=session_id, rag_type=rag_type).first()

        if not chat:
            chat = ChatHistory(session_id=session_id, rag_type=rag_type)
            session.add(chat)
            session.flush()

        existing_messages = session.query(ChatMessage).filter_by(chat_id=chat.id).count()
        new_messages = history[existing_messages:]
        for message in new_messages:
            msg = ChatMessage(
                chat_id = chat.id,
                role = message["role"],
                content = message["content"]
            )
            session.add(msg)
        session.commit()
    finally:
        Session.remove()
    return gr.update(choices=get_chat_histories())

def save_chat_history_arena(history, rag_type, session_id=None):
    """
    Save chat history to the database
    :param history: Chat history
    :param rag_type: RAG type
    :param session_id: Session ID
    """
    session = Session()
    try:
        if session_id is None:
            session_id = generate_session_id()
        chat = session.query(ChatHistoryArena).filter_by(session_id=session_id).first()

        if not chat:
            chat = ChatHistoryArena(session_id=session_id)
            session.add(chat)
            session.flush()

        existing_messages = session.query(ChatMessageArena).filter_by(chat_id=chat.id, rag_type=rag_type).count()
        new_messages = history[existing_messages:]
        for message in new_messages:
            msg = ChatMessageArena(
                chat_id = chat.id,
                rag_type = rag_type,
                role = message["role"],
                content = message["content"]
            )
            session.add(msg)
        session.commit()
    finally:
        Session.remove()
    return gr.update(choices=get_chat_histories_arena())

def get_chat_histories():
    """
    Get chat histories from the database
    :return: Choices for dropdown, Initial value for dropdown
    """
    session = Session()
    try:
        histories = session.query(ChatHistory).order_by(ChatHistory.timestamp.desc()).all()
        choices = [f"{history.rag_type} - {history.session_id}" for history in histories]
        initial_value = choices[0] if choices else None
    finally:
        Session.remove()
    return choices, initial_value

def get_chat_histories_arena():
    """
    Get chat histories from the database
    :return: Choices for dropdown, Initial value for dropdown
    """
    session = Session()
    try:
        histories = session.query(ChatHistoryArena).order_by(ChatHistoryArena.timestamp.desc()).all()
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
        return [], None

    session_id = session_data.get('value') if isinstance(session_data, dict) else session_data
    rag_type, session_id = session_id.split(" - ")
    try:
        chat = session.query(ChatHistory).filter_by(session_id=session_id).first()
        messages = chat.messages

        return [{"role": message.role, "content": message.content} for message in messages], session_id
    finally:
        Session.remove()

def load_chat_history_arena(session_data):
    """
    Load chat history from the database
    :param session_data: Session data
    :return: Chat history
    """
    session = Session()
    if not session_data:
        return [], [], None

    session_id = session_data.get('value') if isinstance(session_data, dict) else session_data
    try:
        chats = session.query(ChatHistoryArena).filter_by(session_id=session_id).all()
        if not chats:
            return [], [], None
        naive_chat = [{"role": message.role, "content": message.content} for chat in chats for message in chat.naive_messages]
        lightrag_chat = [{"role": message.role, "content": message.content} for chat in chats for message in chat.light_messages]

        return naive_chat, lightrag_chat, session_id
    finally:
        Session.remove()

def refresh_histories():
    """
    Refresh chat histories
    :return: Updated choices and value
    """
    choices, value = get_chat_histories()
    return gr.update(choices=choices, value=value)

def refresh_histories_arena():
    """
    Refresh chat histories
    :return: Updated choices and value
    """
    choices, value = get_chat_histories_arena()
    return gr.update(choices=choices, value=value)

def delete_chat(session_data, arena_flag=False):
    """
    Delete chat history from the database
    """
    session = Session()
    if arena_flag:
        if not session_data:
            choices, value = get_chat_histories()
            return [], [], gr.update(choices=choices, value=value)
        session_id = session_data.get('value') if isinstance(session_data, dict) else session_data
        try:
            chat = session.query(ChatHistoryArena).filter_by(session_id=session_id).first()
            if chat:
                session.delete(chat)
                session.commit()
        finally:
            Session.remove()
        choices, value = get_chat_histories_arena()
        return [], [], gr.update(choices=choices, value=value)
    else:
        if not session_data:
            choices, value = get_chat_histories()
            return [], gr.update(choices=choices, value=value)
        rag_type, session_id = session_data.split(" - ")
        try:
            chat = session.query(ChatHistory).filter_by(session_id=session_id).first()
            if chat:
                session.delete(chat)
                session.commit()
        except:
            chat = session.query(ChatHistoryArena).filter_by(session_id=session_id).first()
            if chat:
                session.delete(chat)
                session.commit()
        finally:
            Session.remove()
        choices, value = get_chat_histories()
        return [], gr.update(choices=choices, value=value)

def pdf_viewer(history):
    if history[-1]["role"] == "assistant":
        return True
    return False