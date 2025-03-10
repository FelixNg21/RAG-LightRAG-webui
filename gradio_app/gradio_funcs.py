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
from services.document_loader import DocumentLoader



class FileManager:
    def __init__(self, save_dir: str, document_loader: DocumentLoader, lightrag: LightRagWrapper):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.document_loader = document_loader
        self.lightrag = lightrag

    def save_files(self, files):
        """
        Save files to the data directory
        :param files: List of files
        :return: Message with the files uploaded
        """
        if not files:
            return "No files uploaded", None
        file_paths = []
        for file in files:
            file_path = os.path.join(self.save_dir, os.path.basename(file.name))
            shutil.move(file.name, file_path)
            file_paths.append(file_path)
        return file_paths, None

    def update_files(self):
        """
        Update the list of files in the directory
        :return: CheckboxGroup with the updated files
        """
        updated_files = self.list_files()
        return gr.CheckboxGroup(label="Uploaded Files", choices=updated_files)

    def list_files(self):
        """
        List the files in the data directory
        :return: List of files or message if no files
        """
        files = os.listdir(self.save_dir)
        return files

    def process_files(self, uploaded_files):
        """
        Process the selected files for both NaiveRAG and LightRAG
        :param selected_files: List of selected files
        :return: Message with the files processed
        """
        if not uploaded_files:
            return "No files uploaded"

        try:
            # Processing for NaiveRAG
            self.document_loader.ingest(uploaded_files)

            # Processing for LightRAG
            self.lightrag.ingest(uploaded_files)

        except Exception as e:
            return "Error processing files: " + str(e)

        return "Processed selected files: " + str(uploaded_files)

    def delete_files(self, files):
        """
        Delete files from the data directory
        :param files: List of files
        :return: Message with the files deleted
        """
        if not files:
            return "No files selected"
        for file in files:
            # Delete from NaiveRAG
            self.document_loader.delete_document(file)

            # Delete from LightRAG
            self.lightrag.delete_document(file)

            file_path = os.path.join(self.save_dir, file)
            os.remove(file_path)
        return "Deleted selected files: " + str(files)


class HistoryManager:
    def __init__(self, db_path):
        self.engine = create_engine('sqlite:///' + db_path)
        if not os.path.exists(db_path):
            db.Model.metadata.create_all(self.engine)
        self.SessionFactory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.SessionFactory)

    def save_chat_history(self, history, rag_type, session_id=None):
        """
        Save chat history to the database
        :param history: Chat history
        :rag_type: RAG type
        :session_id: Session ID
        """
        session = self.Session()
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
                    chat_id=chat.id,
                    role=message["role"],
                    content=message["content"]
                )
                session.add(msg)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.Session.remove()
        return gr.update(choices=self.get_chat_histories())

    def save_chat_history_arena(self, history, rag_type, session_id=None):
        """
        Save chat history to the database
        :param history: Chat history
        :param rag_type: RAG type
        :param session_id: Session ID
        """
        session = self.Session()
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
                    chat_id=chat.id,
                    rag_type=rag_type,
                    role=message["role"],
                    content=message["content"]
                )
                session.add(msg)
            session.commit()
        finally:
            self.Session.remove()
        return gr.update(choices=self.get_chat_histories_arena())

    def get_chat_histories(self):
        """
        Get chat histories from the database
        :return: Choices for dropdown, Initial value for dropdown
        """
        session = self.Session()
        try:
            histories = session.query(ChatHistory).order_by(ChatHistory.timestamp.desc()).all()
            choices = [f"{history.rag_type} - {history.session_id}" for history in histories]
            initial_value = choices[0] if choices else None
        finally:
            self.Session.remove()
        return choices, initial_value

    def get_chat_histories_arena(self):
        """
        Get chat histories from the database
        :return: Choices for dropdown, Initial value for dropdown
        """
        session = self.Session()
        try:
            histories = session.query(ChatHistoryArena).order_by(ChatHistoryArena.timestamp.desc()).all()
            choices = [history.session_id for history in histories]
            initial_value = choices[0] if choices else None

        finally:
            self.Session.remove()
        return choices, initial_value

    def load_chat_history(self, session_data):
        """
        Load chat history from the database
        :param session_data: Session data
        :return: Chat history
        """
        session = self.Session()
        if not session_data:
            return [], None

        session_id = session_data.get('value') if isinstance(session_data, dict) else session_data
        rag_type, session_id = session_id.split(" - ")
        try:
            chat = session.query(ChatHistory).filter_by(session_id=session_id).first()
            messages = chat.messages

            return [{"role": message.role, "content": message.content} for message in messages], session_id
        finally:
            self.Session.remove()

    def load_chat_history_arena(self, session_data):
        """
        Load chat history from the database
        :param session_data: Session data
        :return: Chat history
        """
        session = self.Session()
        if not session_data:
            return [], [], None

        session_id = session_data.get('value') if isinstance(session_data, dict) else session_data
        try:
            chats = session.query(ChatHistoryArena).filter_by(session_id=session_id).all()
            if not chats:
                return [], [], None
            naive_chat = [{"role": message.role, "content": message.content} for chat in chats for message in
                          chat.naive_messages]
            lightrag_chat = [{"role": message.role, "content": message.content} for chat in chats for message in
                             chat.light_messages]

            return naive_chat, lightrag_chat, session_id
        finally:
            self.Session.remove()

    def refresh_histories(self):
        """
        Refresh chat histories
        :return: Updated choices and value
        """
        choices, value = self.get_chat_histories()
        return gr.update(choices=choices, value=value)

    def refresh_histories_arena(self):
        """
        Refresh chat histories
        :return: Updated choices and value
        """
        choices, value = self.get_chat_histories_arena()
        return gr.update(choices=choices, value=value)

    def delete_chat(self, session_data, arena_flag=False):
        """
        Delete chat history from the database
        """
        session = self.Session()
        if arena_flag:
            if not session_data:
                choices, value = self.get_chat_histories()
                return [], [], gr.update(choices=choices, value=value)
            session_id = session_data.get('value') if isinstance(session_data, dict) else session_data
            try:
                chat = session.query(ChatHistoryArena).filter_by(session_id=session_id).first()
                if chat:
                    session.delete(chat)
                    session.commit()
            finally:
                self.Session.remove()
            choices, value = self.get_chat_histories_arena()
            return [], [], gr.update(choices=choices, value=value)
        else:
            if not session_data:
                choices, value = self.get_chat_histories()
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
                self.Session.remove()
            choices, value = self.get_chat_histories()
            return [], gr.update(choices=choices, value=value)


class ChatManager:
    def __init__(self, chroma_db, lightrag_instance, history_manager: HistoryManager):
        self.lightrag = lightrag_instance
        self.ollama = OllamaInterface("deepseek-r1e:latest", chroma_db.db)
        self.history_manager = history_manager

    def user(self, user_message, history: list, session_id=None):
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

    def get_context(self, history, user_message, doc_ids=None):
        """
        Get context based on the user message
        :param history: Chat history
        :param user_message: User message
        """
        context = self.ollama.get_context(user_message, doc_ids)
        return context, history, user_message

    def handle_reasoning(self, model_response):
        """
        Parse reasoning from model_response
        :param model_response: Model response
        :return: Model response, Thinking
        """
        if "<think>" in model_response:
            thinking = re.findall(r"<think>.*?</think>", model_response, flags=re.DOTALL)
            model_response = re.sub(r"<think>.*?</think>", "", model_response, flags=re.DOTALL)
            return model_response, thinking
        return model_response, None

    def assistant(self, history: list, user_message, rag_type, context, session_id=None, arena_flag=False):
        """
        Handle assistant response
        :param history: Chat history
        :param user_message: User message
        :param rag_type: RAG type
        :param context: Context from vector search
        :param session_id: Session ID
        :param arena_flag: Arena flag
        """
        try:
            if rag_type == "LightRAG":
                content = self.lightrag.query(user_message, history=history)
            else:
                response = self.ollama.query(user_message, use_context=True, history=history, context=context)
                content = response["message"]["content"]
            content, thinking = self.handle_reasoning(content)
            history.append({"role": "assistant", "content": ""})
            for character in content:
                history[-1]["content"] += character
                # yield history

            if arena_flag:
                self.history_manager.save_chat_history_arena(history, rag_type, session_id)
            else:
                self.history_manager.save_chat_history(history, rag_type, session_id)
            return history
        except Exception as e:
            print(f"Error in assistant: {e}")


def pdf_viewer(history):
    if history[-1]["role"] == "assistant":
        return True
    return False
