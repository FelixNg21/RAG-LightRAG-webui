import gradio as gr
from services.lightrag_wrapper import LightRagWrapper
from services.ollama_interface import OllamaInterface
from services.chroma_db import Database
import time

lightrag = LightRagWrapper(working_dir="lightrag_docs", llm_model_name="deepseek-r1:8b",
                           doc_dir="./data/pdfs-lightrag")
chroma_db = Database(chroma_path="chroma", collection_name="documents")
ollama = OllamaInterface("llama3.1", chroma_db.db)

with gr.Blocks() as chat_app:
    rag_type = gr.Radio(
        choices=["NaiveRAG", "LightRAG"],
        label="Select RAG Type",
        value="NaiveRAG",
    )
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    user_message_state = gr.State("")

    def user(user_message, history: list, rag_type):
        history = history or []
        history = history + [{"role": "user", "content": user_message}]
        print("In user", history)
        return "", history, user_message, rag_type


    def assistant(history: list, user_message, rag_type):
        if rag_type == "LightRAG":
            response = lightrag.query(user_message, False)
        else:
            response = ollama.query(user_message, use_context=False, history=history)
        content = response["message"]["content"]
        # print(content)
        history.append({"role": "assistant", "content": ""})

        for character in content:
            history[-1]["content"] += character
            # time.sleep(0.05)
            yield history

        # return history


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
