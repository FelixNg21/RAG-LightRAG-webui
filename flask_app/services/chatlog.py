from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class ChatLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String, nullable=False)
    user_query = db.Column(db.String, nullable=False)
    chatbot_response = db.Column(db.String, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
