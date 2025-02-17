import gradio as gr
from services.lightrag_wrapper import LightRagWrapper
from services.ollama_interface import OllamaInterface
from services.chroma_db import Database
from services.document_loader import DocumentLoader
from gradio_pdf import PDF
import time
import os
import shutil
import re

lightrag = LightRagWrapper(working_dir="lightrag_docs", llm_model_name="deepseek-r1:8b",
                           doc_dir="./data/pdfs")
chroma_db = Database(chroma_path="chroma", collection_name="documents")
ollama = OllamaInterface("deepseek-r1:8b", chroma_db.db)
document_loader = DocumentLoader(chroma_db.db, collection_name="documents", data_path="data/pdfs")

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
    print(selected_files)
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
    print(history)
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


def pdf_viewer(history):
    if history[-1]["role"] == "assistant":
        return True
    return False


# Gradio App
with gr.Blocks(fill_height=True) as chat_app:
    with gr.Row(scale=1):
        rag_type = gr.Radio(
            choices=["NaiveRAG", "LightRAG"],
            label="Select RAG Type",
            value="NaiveRAG",
        )
    with gr.Row(scale=50):
        with gr.Column():
            file_input = gr.Files(label="Documents")
            file_checkboxes = gr.CheckboxGroup(label="Uploaded Files", choices=list_files())
            process_files_button = gr.Button("Process Files")
            process_files_output = gr.Textbox()

            file_input.change(save_files, inputs=file_input).then(
                update_files, outputs=file_checkboxes
            )
            process_files_button.click(process_files, inputs=[file_checkboxes], outputs=process_files_output)

        with gr.Column():
            chat_log = gr.Chatbot(type="messages", scale=0)
            msg = gr.Textbox(label="Message")
            clear = gr.Button("Clear")
            user_message_state = gr.State("")

        with gr.Column():
            context = gr.State()
            pdf_component = PDF(visible=False)

    msg.submit(
        fn=user,
        inputs=[msg, chat_log, rag_type],
        outputs=[msg, chat_log, user_message_state, rag_type],
        queue=True
    ).then(
        fn=get_context,
        inputs=[chat_log, user_message_state, rag_type],
        outputs=[context, chat_log, user_message_state, rag_type],
        queue=True
    ).then(
        fn=assistant,
        inputs=[chat_log, user_message_state, rag_type, context],
        outputs=chat_log,
        queue=True
    )
    clear.click(lambda: None, None, chat_log, queue=False)

chat_app.launch(server_name="0.0.0.0", server_port=5000, root_path="http://rag.felicks.duckdns.org")
