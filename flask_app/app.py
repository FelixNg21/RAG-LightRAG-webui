from flask import Flask, render_template, url_for
from services.routes import route_api
from flask_htmx import HTMX
import os
from services.chatlog import db

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat_log.db'
db.init_app(app)
app.register_blueprint(route_api)
htmx = HTMX(app)

with app.app_context():
    db.create_all()


@app.route('/')
def home():
    return render_template('index.html', files=os.listdir("data/pdfs"))


@app.route('/chat')
def chat():
    return render_template('chat.html')


@app.route('/models')
def models():
    return render_template('models.html')


@app.context_processor
def inject_dict_for_all_templates():
    nav = [
        {'text': "Home", "url": url_for('home')},
        {'text': "Chat", "url": url_for('chat')},
        {'text': "Models", "url": url_for('models')},
    ]
    return dict(sidebar=nav)

# @app.context_processor
# def inject_dict_for_chat_html():
#     bar = [
#         {'text': "New Chat", "url": url_for('route_api.new_session')},
#     ]
#     return dict(chat_navbar=bar)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
