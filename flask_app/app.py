from flask import Flask, render_template, url_for
from flask_sqlalchemy import SQLAlchemy
from services.routes import route_api
from flask_htmx import HTMX
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)
# flask_app.app_context().push()
# flask_app.secret_key='supersecretkey'
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
        {'text': "Chat", "url": url_for('chat')},
    ]
    return dict(navbar=nav)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
