import os
import gradio as gr
from gradio_pdf import PDF

from app_container import ApplicationContainer
from gradio_ui_adapter import GradioUIAdapter
from gradio_funcs import FileManager, ChatManager, HistoryManager
from services.lightrag_wrapper import LightRagWrapper
from services.utils import generate_session_id
from services.document_loader import DocumentLoader
from services.chroma_db import Database
from dotenv import load_dotenv



load_dotenv()
SAVE_DIR = os.environ.get("DOCUMENT_DIR")
CHROMA_PATH = os.environ.get("CHROMA_PATH")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION")
LIGHTRAG_DIR = os.environ.get("LIGHTRAG_DIR")
LR_INGEST = os.environ.get("LR_INGEST")
LR_GEN = os.environ.get("LR_GEN")
CHAT_LOG_DB = os.environ.get("CHAT_LOG_DB")

#TODO:
# 2. Add ability to talk to selected documents
#       - implemented for NaiveRAG
#       - LightRAG not implemented
# 4. Speed up ingestion of documents
#       - already fast for NaiveRAG
#       - LightRAG inherently slow
# 5. Simplify DocumentLoader to just take a chroma_db
# 6. LightRAG currently has no ability to understand tables from documents

# chroma_db = Database(chroma_path=CHROMA_PATH, collection_name=CHROMA_COLLECTION)
# document_loader = DocumentLoader(db=chroma_db.db, data_path=SAVE_DIR, collection_name=CHROMA_COLLECTION)
# lightrag_wrapper = LightRagWrapper(working_dir=LIGHTRAG_DIR, llm_model_ingest=LR_INGEST, llm_model_gen=LR_GEN, doc_dir=SAVE_DIR)
# file_manager = FileManager(save_dir=SAVE_DIR, document_loader=document_loader, lightrag=lightrag_wrapper)
# history_manager = HistoryManager(db_path=CHAT_LOG_DB)
# chat_manager = ChatManager(chroma_db=chroma_db, lightrag_instance=lightrag_wrapper, history_manager=history_manager)

def create_file_upload_component(ui):
    with gr.Column(scale=2):
        file_input = gr.Files(label="Documents", height="calc(10vh)")
        status = gr.Textbox(label="Status")

        initial_files = ui.file_manager.list_files()
        file_checkboxes = gr.CheckboxGroup(label="Uploaded Files", choices=initial_files)
        delete_files_button = gr.Button(value="Delete Files", variant="stop")

        uploaded_files = gr.State()
        setup_file_handlers(ui, file_input, uploaded_files, status, file_checkboxes, delete_files_button)
        return file_input, status, file_checkboxes, delete_files_button

def setup_file_handlers(ui, file_input, uploaded_files, status, file_checkboxes, delete_files_button):
    file_input.upload(
        ui.save_files, inputs=file_input, outputs=[uploaded_files, file_input],
    ).then(
        ui.process_files, inputs=uploaded_files, outputs=status
    ).success(
        ui.update_files, outputs=file_checkboxes
    )

    delete_files_button.click(
        ui.delete_files,
        inputs=[file_checkboxes],
        outputs=status
    ).success(
        ui.update_files,
        outputs=file_checkboxes
    )

def setup_chat_interface(ui, chat_log, msg_input, rag_type_value, session_id, arena_flag=False):
    user_message_state = gr.State("")
    context = gr.State()

    msg_input.submit(
        fn=ui.user,
        inputs=[msg_input, chat_log, session_id],
        outputs=[msg_input, chat_log, user_message_state, session_id],
        queue=True
    ).then(
        fn=ui.get_context,
        inputs=[chat_log, user_message_state],
        outputs=[context, chat_log, user_message_state],
        queue=True
    ).then(
        fn=ui.assistant,
        inputs=[chat_log, user_message_state, rag_type_value, context, session_id],
        outputs=[chat_log],
        queue=True
    )

def create_gradio_app():
    container = ApplicationContainer()
    ui = GradioUIAdapter(container)

    with gr.Blocks(fill_height=True, fill_width=True) as default:
        with gr.Row(equal_height=True, scale=3):
            session_id = gr.State(None)
            # Select RAG Type
            rag_type = gr.Radio(
                choices=["NaiveRAG", "LightRAG"],
                label="Select RAG Type",
                value="NaiveRAG",
            )

            initial_choice, initial_value = ui.get_chat_histories()
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
            create_file_upload_component(ui)

            with gr.Column(scale=5):
                chat_log = gr.Chatbot(type="messages", height="calc(100vh - 500px)")
                msg = gr.Textbox(label="Message")
                with gr.Row():
                    refresh = gr.Button("Refresh")
                    clear = gr.Button("Clear")
                setup_chat_interface(
                    ui=ui,
                    chat_log=chat_log,
                    msg_input=msg,
                    rag_type_value=rag_type,
                    session_id=session_id
                )

            with gr.Column(visible=False):
                pdf_component = PDF(visible=False)

        default.load(
            fn=ui.load_chat_history,
            inputs=[chat_history_dropdown],
            outputs=[chat_log, session_id]
        )
        chat_history_dropdown.change(
            fn=ui.load_chat_history,
            inputs=[chat_history_dropdown],
            outputs=[chat_log, session_id]
        )

        refresh.click(
            fn=ui.refresh_histories,
            outputs=[chat_history_dropdown]
        ).success(
            fn=ui.load_chat_history,
            inputs=[chat_history_dropdown],
            outputs=[chat_log, session_id]
        )

        delete_btn.click(
            fn=ui.delete_chat,
            inputs=[chat_history_dropdown],
            outputs=[chat_log, chat_history_dropdown]
        )

        clear.click(lambda: (None, generate_session_id()), None, [chat_log, session_id], queue=False)

    # Arena
    with gr.Blocks(fill_height=True, fill_width=True) as arena:
        pass
        # arena_flag = gr.State(True)
        # session_id_arena = gr.State(None)
        # with gr.Row(equal_height=True, scale=1):
        #     initial_choice, initial_value = get_chat_histories_arena()
        #     # Select Chat History
        #     with gr.Column(scale=10):
        #         chat_history_dropdown = gr.Dropdown(
        #             choices=initial_choice,
        #             value=initial_value,
        #             label="Chat History",
        #             interactive=True,
        #             allow_custom_value=False
        #         )
        #     # Delete Chat
        #     with gr.Column(scale=1):
        #         delete_btn = gr.Button("🗑Delete Chat", variant="stop")
        #
        # with gr.Row(scale=50, equal_height=True):
        #     with gr.Column(scale=1):
        #         file_input = gr.Files(label="Documents")
        #         file_checkboxes = gr.CheckboxGroup(label="Uploaded Files", choices=list_files())
        #         process_files_button = gr.Button("Process Files")
        #         status = gr.Textbox()
        #
        #     with gr.Column(scale=10):
        #         with gr.Row():
        #             with gr.Column():
        #                 gr.Markdown("### NaiveRAG")
        #                 chat_log_naive = gr.Chatbot(type="messages", label="NaiveRAG", height="calc(100vh - 500px)")
        #             with gr.Column():
        #                 gr.Markdown("### LightRAG")
        #                 chat_log_light = gr.Chatbot(type="messages", label="LightRAG", height="calc(100vh - 500px)")
        #
        #         msg = gr.Textbox(label="Message")
        #         with gr.Row():
        #             refresh = gr.Button("Refresh")
        #             clear = gr.Button("Clear")
        #         user_message_state = gr.State("")
        #     with gr.Column(visible=False): # Hidden, intended for showing PDF context
        #         context = gr.State()
        #         pdf_component = PDF(visible=False)
        #
        # file_input.change(
        #     save_files, inputs=file_input).then(
        #     update_files, outputs=file_checkboxes
        # )
        # process_files_button.click(process_files, inputs=[file_checkboxes], outputs=status)
        #
        # arena.load(
        #     fn=load_chat_history_arena,
        #     inputs=[chat_history_dropdown],
        #     outputs=[chat_log_naive, chat_log_light, session_id_arena]
        # )
        #
        # chat_history_dropdown.change(
        #     fn=load_chat_history_arena,
        #     inputs=[chat_history_dropdown],
        #     outputs=[chat_log_naive, chat_log_light, session_id_arena]
        # )
        #
        # refresh.click(
        #     fn=refresh_histories_arena,
        #     outputs=[chat_history_dropdown]
        # ).success(
        #     fn=lambda x: load_chat_history_arena(x),
        #     inputs=[chat_history_dropdown],
        #     outputs=[chat_log_naive, chat_log_light, session_id_arena]
        # )
        #
        # delete_btn.click(
        #     fn=delete_chat,
        #     inputs=[chat_history_dropdown, arena_flag],
        #     outputs=[chat_log_naive, chat_log_light, chat_history_dropdown]
        # )
        #
        # msg.submit(
        #     fn=user,
        #     inputs=[msg, chat_log_naive, session_id_arena],
        #     outputs=[msg, chat_log_naive, user_message_state, session_id_arena],
        #     queue=True
        # ).then(
        #     fn=get_context,
        #     inputs=[chat_log_naive, user_message_state],
        #     outputs=[context, chat_log_naive, user_message_state],
        #     queue=True
        # ).then(
        #     fn=assistant,
        #     inputs=[chat_log_naive, user_message_state, gr.State("NaiveRAG"), context, session_id_arena, arena_flag],
        #     outputs=[chat_log_naive],
        #     queue=True
        # )
        # msg.submit(
        #     fn=user,
        #     inputs=[msg, chat_log_light, session_id_arena],
        #     outputs=[msg, chat_log_light, user_message_state, session_id_arena],
        #     queue=True
        # ).then(
        #     fn=get_context,
        #     inputs=[chat_log_light, user_message_state],
        #     outputs=[context, chat_log_light, user_message_state],
        #     queue=True
        # ).then(
        #     fn=assistant,
        #     inputs=[chat_log_light, user_message_state, gr.State("LightRAG"), context, session_id_arena, arena_flag],
        #     outputs=[chat_log_light],
        #     queue=True
        # )
        #
        # clear.click(lambda: (None, None, generate_session_id()),
        #             inputs =None,
        #             outputs= [chat_log_naive, chat_log_light, session_id_arena],
        #             queue=False
        # )

    return gr.TabbedInterface([default, arena], tab_names=["Default", "Arena"], title="RAG Chatbot")

if __name__ == "__main__":
    chat_app = create_gradio_app()
    # chat_app.launch(server_name="0.0.0.0", server_port=5000, root_path="https://rag.felicks.duckdns.org", ssl_verify=False)
    chat_app.launch(debug=True)
