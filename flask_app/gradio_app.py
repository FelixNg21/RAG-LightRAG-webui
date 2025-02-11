import gradio as gr
from services.lightrag_wrapper import LightRagWrapper
from services.ollama_interface import OllamaInterface
from services.chroma_db import Database
from services.document_loader import DocumentLoader
import time
import os
import shutil

lightrag = LightRagWrapper(working_dir="lightrag_docs", llm_model_name="deepseek-r1:8b",
                           doc_dir="./data/pdfs-lightrag")
chroma_db = Database(chroma_path="chroma", collection_name="documents")
ollama = OllamaInterface("llama3.1", chroma_db.db)
document_loader = DocumentLoader(chroma_db.db, collection_name="documents", data_path="data/pdfs")

SAVE_DIR = "data/pdfs"
os.makedirs(SAVE_DIR, exist_ok=True)


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


#TODO: Implement file processing
def process_files(selected_files):
    if not selected_files:
        return "No files selected"
    try:
        documents = document_loader.load_documents(selected_files)
        chunks = document_loader.split_documents(documents)
        document_loader.add_to_chroma(chunks)

    except Exception as e:
        return "Error processing files: " + str(e)

    return "Files processed selected files: " + str(selected_files)


def user(user_message, history: list, rag_type):
    history = history or []
    history = history + [{"role": "user", "content": user_message}]
    return "", history, user_message, rag_type


def assistant(history: list, user_message, rag_type):
    if rag_type == "LightRAG":
        response = lightrag.query(user_message, False)
    else:
        response = ollama.query(user_message, use_context=True, history=history)
    content = response["message"]["content"]
    # print(content)
    history.append({"role": "assistant", "content": ""})

    for character in content:
        history[-1]["content"] += character
        # time.sleep(0.05)
        yield history


with gr.Blocks(fill_width=True, fill_height=True) as chat_app:
    with gr.Row():
        with gr.Column(scale=1):
            file_checkboxes = gr.CheckboxGroup(label="Uploaded Files", choices=list_files())
            process_files_button = gr.Button("Process Files")
            process_files_output = gr.Textbox()
            file_input = gr.Files(label="Documents")

            file_input.change(save_files, inputs=file_input).then(
                update_files, outputs=file_checkboxes
            )
            process_files_button.click(process_files, inputs=[file_checkboxes], outputs=process_files_output)

        with gr.Column(scale=4):
            rag_type = gr.Radio(
                choices=["NaiveRAG", "LightRAG"],
                label="Select RAG Type",
                value="NaiveRAG",
            )
            chatbot = gr.Chatbot(type="messages", scale=0)
            msg = gr.Textbox()
            clear = gr.Button("Clear")
            user_message_state = gr.State("")

    msg.submit(
        fn=user,
        inputs=[msg, chatbot, rag_type],
        outputs=[msg, chatbot, user_message_state, rag_type],
        queue=True,
    ).then(
        fn=assistant,
        inputs=[chatbot, user_message_state],
        outputs=chatbot,
        queue=True,
    )
    clear.click(lambda: None, None, chatbot, queue=False)

chat_app.launch()
