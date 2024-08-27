from flask import Flask, render_template, url_for
from flask_sqlalchemy import SQLAlchemy
from app.api.routes import route_api
from flask_htmx import HTMX
import os

app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
# db = SQLAlchemy(app)
# app.app_context().push()
# app.secret_key='supersecretkey'
app.register_blueprint(route_api)
htmx = HTMX(app)


@app.route('/')
def home():
    return render_template('index.html', files=os.listdir("data/pdfs"))

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.context_processor
def inject_dict_for_all_templates():
    nav = [
        {'text': "Home", "url": url_for('home')},
        # {'text': "Upload", "url": url_for('upload')},
        {'text': "Chat", "url": url_for('chat')},
    ]
    return dict(navbar=nav)

if __name__ == "__main__":
    app.run(debug=True)
