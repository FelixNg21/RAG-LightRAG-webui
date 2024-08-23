from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from app.api.routes import route_api
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)
app.app_context().push()
app.secret_key='supersecretkey'
app.register_blueprint(route_api)


@app.route('/')
def home():
    return render_template('index.html', files=os.listdir("data/pdfs"))

if __name__ == "__main__":
    app.run(debug=True)
