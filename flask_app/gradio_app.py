import gradio as gr
from gradio_pdf import PDF
import os

from gradio_funcs import save_files, update_files, process_files, get_chat_histories, load_chat_history, \
    delete_chat, list_files, user, get_context, assistant, refresh_histories


# Gradio App
with gr.Blocks(fill_height=True) as chat_app:
    with gr.Row(equal_height=True):
        rag_type = gr.Radio(
            choices=["NaiveRAG", "LightRAG"],
            label="Select RAG Type",
            value="NaiveRAG",
        )
        initial_choice, initial_value = get_chat_histories()
        with gr.Column(scale=15):
            chat_history_dropdown = gr.Dropdown(
                choices=initial_choice,
                value=initial_value,
                label="Chat History",
                interactive=True,
                allow_custom_value=False
            )
        with gr.Column(scale=1):
            delete_btn = gr.Button("🗑Delete Chat", variant="stop")

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
            with gr.Row():
                refresh = gr.Button("Refresh")
                clear = gr.Button("Clear")
            user_message_state = gr.State("")

        with gr.Column():
            context = gr.State()
            pdf_component = PDF(visible=False)

    chat_history_dropdown.change(
        fn=load_chat_history,
        inputs=[chat_history_dropdown],
        outputs=[chat_log]
    )
    refresh.click(
        fn=refresh_histories,
        outputs=[chat_history_dropdown]
    ).then(
        fn=load_chat_history,
        inputs=[chat_history_dropdown],
        outputs=[chat_log]
    )
    delete_btn.click(
        fn=delete_chat,
        inputs=[chat_history_dropdown],
        outputs=[chat_log, chat_history_dropdown]
    )
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
        outputs=[chat_log],
        queue=True
    )

    clear.click(lambda: None, None, chat_log, queue=False)

# chat_app.launch(server_name="0.0.0.0", server_port=5000, root_path="https://rag.felicks.duckdns.org", ssl_verify=False)
chat_app.launch(debug=True)