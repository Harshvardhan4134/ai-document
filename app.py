from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_session import Session
import os
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from extract_text import extract_text_from_pdf, extract_text_from_xlsx, extract_text_from_json
from openai_utils import generate_embeddings, generate_summary, answer_question
import pinecone
from pinecone import Pinecone, ServerlessSpec
import psycopg2
import config  # Ensure `config.py` has correct DB credentials

app = Flask(__name__)

# ✅ Flask Session Configuration
app.secret_key = "your_secret_key_here"
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# ✅ File Upload Config
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESSED_JSON'] = 'processed_data.json'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'xlsx', 'txt', 'json'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ✅ Connect to PostgreSQL
try:
    conn = psycopg2.connect(
        host=config.DB_HOST,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD
    )
    conn.autocommit = True  
    cursor = conn.cursor()
    print("✅ Database connected successfully!")
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    exit(1)

# ✅ Ensure users table exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(50) UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
""")

# ✅ Initialize Pinecone
pinecone_api_key = "pcsk_JcF8a_DLvNLFhfUk7GrVkiV6ueX9xKbxr9vW8nLij5q8fpD1ZM2VABtLtiWvLTCd3RbFP"
index_name = "document-embeddings"
pc = Pinecone(api_key=pinecone_api_key)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)
print(f"✅ Pinecone index '{index_name}' initialized successfully!")

# ✅ Helper Function: Check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.before_request
def require_login():
    allowed_routes = ['login', 'static']  # Allow login page & static files
    if "user" not in session and request.endpoint not in allowed_routes:
        return redirect(url_for('login'))


# ✅ Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get("username")
        password = request.form.get("password")

        print(f"🔹 Received login request for: {username}")

        cursor.execute("ROLLBACK")  # Prevent transaction errors
        cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        print(f"🔹 Database returned: {user}")  # ✅ Debug print

        if user:
            db_password = user[0]
            if password == db_password:
                print(f"✅ Login successful for {username}")
                session["user"] = username
                return redirect(url_for("index_page"))
            else:
                print(f"❌ Wrong password for {username}")
                return render_template("login.html", error="Invalid username or password")
        else:
            print(f"❌ User not found: {username}")
            return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")  



# ✅ Logout Route
@app.route('/logout')
def logout():
    session.pop("user", None)  
    return redirect(url_for("login"))  

# ✅ Home Page (Redirects to login if not authenticated)
@app.route('/')
def index_page():
    if "user" not in session:  
        print("❌ User not logged in. Redirecting to login...")
        return redirect(url_for('login'))  # ✅ Fix: Redirects to `/login`
    return render_template('index.html')


# ✅ Upload Route (Protected)
@app.route('/upload', methods=['POST'])
def upload_file():
    if "user" not in session:
        return jsonify({"error": "Unauthorized access. Please log in."}), 403

    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400

        files = request.files.getlist('files')
        uploaded_files, summaries, json_data = [], [], []

        # ✅ Clear existing Pinecone index before adding new files
        index.delete(delete_all=True)
        print("✅ Old embeddings deleted. Index refreshed.")

        for file in files:
            if file.filename == '':
                continue
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

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
                    continue

                # ✅ Generate summary
                summary = generate_summary(extracted_text)
                print(f"📄 Summary generated: {summary}")

                # ✅ Generate embeddings
                embeddings = generate_embeddings(extracted_text)
                if embeddings is None:
                    print(f"❌ Error: Embeddings could not be generated for {filename}")
                    continue  

                # ✅ Store embeddings in Pinecone
                index.upsert(vectors=[(filename, embeddings, {"text": extracted_text})])

                # ✅ Store extracted data
                doc_data = {
                    "file_name": filename,
                    "date_uploaded": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "extracted_text": extracted_text,
                    "summary": summary,
                    "metadata": {
                        "file_type": filename.split(".")[-1],
                        "file_size": os.path.getsize(filepath)
                    }
                }
                json_data.append(doc_data)
                summaries.append({"filename": filename, "summary": summary})
                uploaded_files.append(filename)

        # ✅ Save extracted data to JSON file
        with open(app.config['PROCESSED_JSON'], "w", encoding="utf-8") as json_file:
            json.dump(json_data, json_file, ensure_ascii=False, indent=4)

        if not uploaded_files:
            return jsonify({"error": "No valid files uploaded"}), 400

        return jsonify({"message": "Files uploaded successfully", "files": uploaded_files, "summaries": summaries}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    print(f"📂 Extracted Text: {extracted_text[:500]}")  # Debug: Print first 500 chars


# ✅ Ask Question (Protected)
@app.route('/ask', methods=['POST'])
def ask_question():
    if "user" not in session:
        return jsonify({"error": "Unauthorized access. Please log in."}), 403

    try:
        data = request.json
        question = data.get('question')

        if not question:
            return jsonify({"error": "Question is required"}), 400

        print(f"🔍 Received Question: {question}")  # Debug print

        # ✅ Load extracted text from JSON storage
        with open(app.config['PROCESSED_JSON'], "r", encoding="utf-8") as f:
            stored_data = json.load(f)

        # ✅ Combine all extracted document texts
        combined_text = "\n\n".join([doc["extracted_text"][:1000] for doc in stored_data])

        # ✅ If no document data, fallback to general AI response
        if not combined_text.strip():
            combined_text = "No document found. Use general knowledge."

        print(f"📄 Context Sent to AI: {combined_text[:500]}")  # Debugging

        # ✅ Call OpenAI with extracted document text
        answer = answer_question(question, combined_text)

        print(f"✅ AI Answer: {answer}")  # Debug print

        return jsonify({"answer": answer}), 200

    except Exception as e:
        print(f"❌ Error in ask_question: {e}")  # Debug error print
        return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)
