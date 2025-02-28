import gradio as gr
from gradio_pdf import PDF

from gradio_funcs import save_files, update_files, process_files, get_chat_histories, load_chat_history, \
    delete_chat, list_files, user, get_context, assistant, refresh_histories, load_chat_history_arena, get_chat_histories_arena, \
    refresh_histories_arena
from services.utils import generate_session_id

#TODO:
# 1. Add button to delete files (from uploaded files, knowledge base [lightrag, naiverag])
# 3. Refactor gradio_funcs and gradio_funcs_arena.

# Gradio App
with gr.Blocks(fill_height=True) as default:
    with gr.Row(equal_height=True, scale=1):
        session_id = gr.State(None)
        # Select RAG Type
        rag_type = gr.Radio(
            choices=["NaiveRAG", "LightRAG"],
            label="Select RAG Type",
            value="NaiveRAG",
        )

        initial_choice, initial_value = get_chat_histories()
        # Select Chat History
        with gr.Column(scale=10):
            chat_history_dropdown = gr.Dropdown(
                choices=initial_choice,
                value=initial_value,
                label="Chat History",
                interactive=True,
                allow_custom_value=False
            )
        # Delete Chat
        with gr.Column(scale=1):
            delete_btn = gr.Button("🗑Delete Chat", variant="stop")

    with gr.Row(scale=50, equal_height=True):
        with gr.Column(scale=1):
            file_input = gr.Files(label="Documents")
            file_checkboxes = gr.CheckboxGroup(label="Uploaded Files", choices=list_files())
            process_files_button = gr.Button("Process Files")
            process_files_output = gr.Textbox()

            file_input.change(save_files, inputs=file_input).then(
                update_files, outputs=file_checkboxes
            )
            process_files_button.click(process_files, inputs=[file_checkboxes], outputs=process_files_output)

        with gr.Column(scale=5):
            chat_log = gr.Chatbot(type="messages")
            msg = gr.Textbox(label="Message")
            with gr.Row():
                refresh = gr.Button("Refresh")
                clear = gr.Button("Clear")
            user_message_state = gr.State("")

        with gr.Column(visible=False):
            context = gr.State()
            pdf_component = PDF(visible=False)

    default.load(
        fn=load_chat_history,
        inputs=[chat_history_dropdown],
        outputs=[chat_log, session_id]
    )

    chat_history_dropdown.change(
        fn=load_chat_history,
        inputs=[chat_history_dropdown],
        outputs=[chat_log, session_id]
    )

    refresh.click(
        fn=refresh_histories,
        outputs=[chat_history_dropdown]
    ).success(
        fn=lambda x: load_chat_history(x),
        inputs=[chat_history_dropdown],
        outputs=[chat_log, session_id]
    )

    delete_btn.click(
        fn=delete_chat,
        inputs=[chat_history_dropdown],
        outputs=[chat_log, chat_history_dropdown]
    )

    msg.submit(
        fn=user,
        inputs=[msg, chat_log, session_id],
        outputs=[msg, chat_log, user_message_state, session_id],
        queue=True
    ).then(
        fn=get_context,
        inputs=[chat_log, user_message_state],
        outputs=[context, chat_log, user_message_state],
        queue=True
    ).then(
        fn=assistant,
        inputs=[chat_log, user_message_state, rag_type, context, session_id],
        outputs=[chat_log],
        queue=True
    )

    clear.click(lambda: (None, generate_session_id()), None, [chat_log, session_id], queue=False)

# Arena
with gr.Blocks(fill_height=True) as arena:
    # pass
    arena_flag = gr.State(True)
    session_id_arena = gr.State(None)
    with gr.Row(equal_height=True, scale=1):
        initial_choice, initial_value = get_chat_histories_arena()
        # Select Chat History
        with gr.Column(scale=10):
            chat_history_dropdown = gr.Dropdown(
                choices=initial_choice,
                value=initial_value,
                label="Chat History",
                interactive=True,
                allow_custom_value=False
            )
        # Delete Chat
        with gr.Column(scale=1):
            delete_btn = gr.Button("🗑Delete Chat", variant="stop")

    with gr.Row(scale=50, equal_height=True):
        with gr.Column(scale=1):
            file_input = gr.Files(label="Documents")
            file_checkboxes = gr.CheckboxGroup(label="Uploaded Files", choices=list_files())
            process_files_button = gr.Button("Process Files")
            process_files_output = gr.Textbox()

        with gr.Column(scale=10):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### NaiveRAG")
                    chat_log_naive = gr.Chatbot(type="messages", label="NaiveRAG", height="calc(100vh - 500px)")
                with gr.Column():
                    gr.Markdown("### LightRAG")
                    chat_log_light = gr.Chatbot(type="messages", label="LightRAG", height="calc(100vh - 500px)")

            msg = gr.Textbox(label="Message")
            with gr.Row():
                refresh = gr.Button("Refresh")
                clear = gr.Button("Clear")
            user_message_state = gr.State("")
        with gr.Column(visible=False): # Hidden, intended for showing PDF context
            context = gr.State()
            pdf_component = PDF(visible=False)

    file_input.change(save_files, inputs=file_input).then(
        update_files, outputs=file_checkboxes
    )
    process_files_button.click(process_files, inputs=[file_checkboxes], outputs=process_files_output)

    arena.load(
        fn=load_chat_history_arena,
        inputs=[chat_history_dropdown],
        outputs=[chat_log_naive, chat_log_light, session_id_arena]
    )

    chat_history_dropdown.change(
        fn=load_chat_history_arena,
        inputs=[chat_history_dropdown],
        outputs=[chat_log_naive, chat_log_light, session_id_arena]
    )

    refresh.click(
        fn=refresh_histories_arena,
        outputs=[chat_history_dropdown]
    ).success(
        fn=lambda x: load_chat_history_arena(x),
        inputs=[chat_history_dropdown],
        outputs=[chat_log_naive, chat_log_light, session_id_arena]
    )

    delete_btn.click(
        fn=delete_chat,
        inputs=[chat_history_dropdown, arena_flag],
        outputs=[chat_log_naive, chat_log_light, chat_history_dropdown]
    )

    msg.submit(
        fn=user,
        inputs=[msg, chat_log_naive, session_id_arena],
        outputs=[msg, chat_log_naive, user_message_state, session_id_arena],
        queue=True
    ).then(
        fn=get_context,
        inputs=[chat_log_naive, user_message_state],
        outputs=[context, chat_log_naive, user_message_state],
        queue=True
    ).then(
        fn=assistant,
        inputs=[chat_log_naive, user_message_state, gr.State("NaiveRAG"), context, session_id_arena, arena_flag],
        outputs=[chat_log_naive],
        queue=True
    )
    msg.submit(
        fn=user,
        inputs=[msg, chat_log_light, session_id_arena],
        outputs=[msg, chat_log_light, user_message_state, session_id_arena],
        queue=True
    ).then(
        fn=get_context,
        inputs=[chat_log_light, user_message_state],
        outputs=[context, chat_log_light, user_message_state],
        queue=True
    ).then(
        fn=assistant,
        inputs=[chat_log_light, user_message_state, gr.State("LightRAG"), context, session_id_arena, arena_flag],
        outputs=[chat_log_light],
        queue=True
    )

    clear.click(lambda: (None, None, generate_session_id()),
                inputs =None,
                outputs= [chat_log_naive, chat_log_light, session_id_arena],
                queue=False
    )

chat_app = gr.TabbedInterface([default, arena], tab_names=["Default", "Arena"], title="RAG Chatbot")

if __name__ == "__main__":
    chat_app.launch(server_name="0.0.0.0", server_port=5000, root_path="https://rag.felicks.duckdns.org", ssl_verify=False)
    # chat_app.launch(debug=True)