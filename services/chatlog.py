from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
# from pyarrow.array import nulls
# from pydantic_core.core_schema import nullable_schema

db = SQLAlchemy()


class ChatHistory(db.Model):
    __tablename__ = 'chat_history'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), nullable=False)
    rag_type = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    messages = db.relationship('ChatMessage',
                               # back_populates='chat',
                               cascade='all, delete-orphan')

class ChatMessage(db.Model):
    __tablename__ = 'chat_message'
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat_history.id'), nullable=False)
    role = db.Column(db.String(10), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    # chat = db.relationship('ChatHistory', back_populates='messages')

class ChatHistoryArena(db.Model):
    __tablename__ = 'chat_history_arena'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    naive_messages = db.relationship(
        'ChatMessageArena',
        primaryjoin="and_(ChatMessageArena.chat_id==ChatHistoryArena.id, ChatMessageArena.rag_type=='NaiveRAG')",
        overlaps='light_messages',
        cascade='all, delete-orphan')
    light_messages = db.relationship(
        'ChatMessageArena',
        primaryjoin="and_(ChatMessageArena.chat_id==ChatHistoryArena.id, ChatMessageArena.rag_type=='LightRAG')",
        overlaps='naive_messages',
        cascade='all, delete-orphan')

class ChatMessageArena(db.Model):
    __tablename__ = 'chat_message_arena'
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat_history_arena.id'), nullable=False)
    role = db.Column(db.String(10), nullable=False)
    rag_type = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
