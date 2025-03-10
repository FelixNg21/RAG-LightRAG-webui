import gradio as gr

class GradioUIAdapter:
    def __init__(self, container):
        self.container = container
        self.file_manager = container.get("file_manager")
        self.chat_manager = container.get("chat_manager")
        self.history_manager = container.get("history_manager")

    # File handling methods
    def save_files(self, files):
        return self.file_manager.save_files(files)

    def process_files(self, uploaded_files):
        return self.file_manager.process_files(uploaded_files)

    def update_files(self):
        files = self.file_manager.list_files()
        return gr.update(choices=files)

    def delete_files(self, file_checkboxes):
        self.file_manager.delete_files(file_checkboxes)

    # Chat methods
    def user(self, msg, chat_log, session_id):
        return self.chat_manager.user(msg, chat_log, session_id)

    def get_context(self, chat_log, user_message):
        return self.chat_manager.get_context(chat_log, user_message)

    def assistant(self, chat_log, user_message, rag_type, context, session_id):
        return self.chat_manager.assistant(chat_log, user_message, rag_type, context, session_id)

    # History methods
    def get_chat_histories(self):
        return self.history_manager.get_chat_histories()

    def load_chat_history(self, session_id):
        return self.history_manager.load_chat_history(session_id)

    def refresh_histories(self):
        choices, value = self.history_manager.get_chat_histories()
        return gr.update(choices=choices, value=value)

    def delete_chat(self, session_id):
        return self.history_manager.delete_chat(session_id)