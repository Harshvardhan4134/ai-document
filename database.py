from sqlalchemy import create_engine, Column, Integer, String, Text, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB

# Database connection URL
DATABASE_URL = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative models
Base = declarative_base()

# Define the Folders table
class Folder(Base):
    __tablename__ = "folders"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)

# Define the Documents table
class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    text = Column(Text)
    embeddings = Column(LargeBinary)  # Store embeddings as binary data
    folder_id = Column(Integer, index=True)

# Create all tables in the database
Base.metadata.create_all(bind=engine)

# Utility functions for database operations
def save_to_database(filename, text, embeddings, folder_id):
    db = SessionLocal()
    try:
        db_document = Document(filename=filename, text=text, embeddings=embeddings, folder_id=folder_id)
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
    finally:
        db.close()

def create_folder(folder_name):
    db = SessionLocal()
    try:
        db_folder = Folder(name=folder_name)
        db.add(db_folder)
        db.commit()
        db.refresh(db_folder)
    finally:
        db.close()

def get_folders():
    db = SessionLocal()
    try:
        folders = db.query(Folder).all()
        return folders
    finally:
        db.close()

def get_documents_in_folder(folder_id):
    db = SessionLocal()
    try:
        documents = db.query(Document).filter(Document.folder_id == folder_id).all()
        return documents
    finally:
        db.close()

def query_database(query):
    """
    Query the database for documents relevant to the user's question.
    """
    db = SessionLocal()
    try:
        # Search for documents containing the query in their text
        relevant_docs = db.query(Document).filter(Document.text.ilike(f"%{query}%")).all()
        return [doc.text for doc in relevant_docs]
    finally:
        db.close()