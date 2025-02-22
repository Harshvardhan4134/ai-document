from flask import Flask, request, jsonify, render_template
import os
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from extract_text import extract_text_from_pdf, extract_text_from_xlsx, extract_text_from_json
from openai_utils import generate_embeddings, generate_summary, answer_question
import pinecone
from pinecone import Pinecone, ServerlessSpec
from database import create_folder
from datetime import timedelta
from flask_jwt_extended import JWTManager, create_access_token


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESSED_JSON'] = 'processed_data.json'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'xlsx', 'txt', 'json'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

 #Dummy users (Replace with DB storage later)
jwt = JWTManager(app)  # ✅ This must be inside your app

USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "user"}
}

# Initialize Pinecone
pinecone_api_key = "YOUR_PINECONEAPI_KEY"
index_name = "document-embeddings"

pc = Pinecone(api_key=pinecone_api_key)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)
print(f"✅ Pinecone index '{index_name}' initialized successfully!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 🔹 User Login (Returns JWT Token)
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if username in USERS and USERS[username]["password"] == password:
        user_role = USERS[username]["role"]
        access_token = create_access_token(identity={"username": username, "role": user_role}, expires_delta=timedelta(hours=1))
        return jsonify({"access_token": access_token, "role": user_role}), 200
    else:
        return jsonify({"error": "Invalid username or password"}), 401

# 🔹 Serve Login Page (New!)
@app.route('/login_page')
def login_page():
    return render_template("login.html")

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/create_folder', methods=['POST'])
def create_folder_route():
    folder_name = request.json.get('folder_name')
    if not folder_name:
        return jsonify({"error": "Folder name is required"}), 400
    create_folder(folder_name)
    return jsonify({"message": f"Folder '{folder_name}' created successfully"}), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400

        files = request.files.getlist('files')
        uploaded_files = []
        summaries = []
        json_data = []

        for file in files:
            if file.filename == '':
                continue
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Extract text based on file type
                extracted_text = None
                if filename.endswith(".pdf"):
                    extracted_text = extract_text_from_pdf(filepath)
                elif filename.endswith(".xlsx"):
                    extracted_text = extract_text_from_xlsx(filepath)
                elif filename.endswith(".json"):
                    extracted_text = extract_text_from_json(filepath)
                else:
                    return jsonify({"error": f"Unsupported file type: {filename}"}), 400

                if not extracted_text.strip():
                    print(f"❌ No text extracted from {filename}. Skipping embedding generation.")
                    continue  # Skip instead of returning an error for one bad file

                # Generate summary before storing
                summary = generate_summary(extracted_text)

                # Store extracted data in JSON format
                doc_data = {
                    "file_name": filename,
                    "date_uploaded": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "extracted_text": extracted_text,
                    "summary": summary,  # Store summary with extracted data
                    "metadata": {
                        "file_type": filename.split(".")[-1],
                        "file_size": os.path.getsize(filepath)
                    }
                }
                json_data.append(doc_data)

                # Append summary to the response
                summaries.append({"filename": filename, "summary": summary})
                uploaded_files.append(filename)

        # Save extracted data to JSON file
        with open(app.config['PROCESSED_JSON'], "w", encoding="utf-8") as json_file:
            json.dump(json_data, json_file, ensure_ascii=False, indent=4)

        # Store document embeddings in Pinecone
        for doc in json_data:
            filename = doc["file_name"]
            extracted_text = doc["extracted_text"]
            embeddings = generate_embeddings(extracted_text)

            if embeddings is None:
                print(f"❌ Error: Embeddings could not be generated for {filename}")
                continue  # Skip instead of failing the entire process

            print(f"✅ Upserting embeddings for {filename} with {len(embeddings)} dimensions")
            index.upsert(vectors=[(filename, embeddings, {"text": extracted_text})])

        if not uploaded_files:
            return jsonify({"error": "No valid files uploaded"}), 400

        return jsonify({"message": "Files uploaded successfully", "files": uploaded_files, "summaries": summaries}), 200

    except Exception as e:
        print(f"❌ Error in upload_file: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        question = data.get('question')
        if not question:
            return jsonify({"error": "Question is required"}), 400

        # Generate query embeddings
        query_embedding = generate_embeddings(question)

        if query_embedding is None:
            return jsonify({"error": "Failed to generate embeddings for the question"}), 500

        print(f"🔍 Querying Pinecone index with question: {question}")
        results = index.query(vector=[query_embedding], top_k=10, include_metadata=True)

        if not results or "matches" not in results or len(results["matches"]) == 0:
            return jsonify({"error": "No relevant documents found"}), 404

        # Filter only highly relevant documents (score > 0.75)
        filtered_docs = [
            {"filename": match["id"], "text": match["metadata"]["text"], "score": match["score"]}
            for match in results["matches"]
            if match["score"] > 0.75
        ]

        if not filtered_docs:
            return jsonify({"error": "No highly relevant documents found"}), 404

        # Combine relevant text
        combined_text = "\n\n".join([doc["text"][:1000] for doc in filtered_docs])

        # Use GPT-4 to generate an answer based on retrieved documents
        answer = answer_question(question, combined_text)

        return jsonify({
            "answer": answer,
            "references": [
                {"filename": doc["filename"], "score": round(doc["score"], 2)}
                for doc in filtered_docs
            ]
        }), 200

    except Exception as e:
        print(f"❌ Error in ask_question: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
