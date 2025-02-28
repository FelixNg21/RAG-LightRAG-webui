from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
# from pyarrow.array import nulls
# from pydantic_core.core_schema import nullable_schema

db = SQLAlchemy()


class ChatHistory(db.Model):
    __tablename__ = 'chat_history_arena'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    naive_messages = db.relationship(
        'ChatMessage',
        primaryjoin="and_(ChatMessage.chat_id==ChatHistory.id, ChatMessage.rag_type=='NaiveRAG')",
        overlaps='light_messages',
        cascade='all, delete-orphan')
    light_messages = db.relationship(
        'ChatMessage',
        primaryjoin="and_(ChatMessage.chat_id==ChatHistory.id, ChatMessage.rag_type=='LightRAG')",
        overlaps='naive_messages',
        cascade='all, delete-orphan')

class ChatMessage(db.Model):
    __tablename__ = 'chat_message'
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat_history.id'), nullable=False)
    role = db.Column(db.String(10), nullable=False)
    rag_type = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    # chat = db.relationship('ChatHistory', back_populates='messages')
