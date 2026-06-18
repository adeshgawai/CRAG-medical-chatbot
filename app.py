import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from src.graph import build_graph
from src.models import db, User

load_dotenv()

crag_app = build_graph()

# Initialize the Flask application
app = Flask(__name__)

# Configure SQLAlchemy and Flask-Login
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-12345')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configuring the database URI (compatible with Supabase PostgreSQL and SQLite)
db_url = os.environ.get('DATABASE_URL', 'sqlite:///medical_chatbot.db')
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = db_url

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = 'landing'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create tables if they do not exist
with app.app_context():
    db.create_all()


# --- Global Variables ---
# These will be loaded once when the app starts
# retriever = None
# llm = None
# VECTOR_STORE_PATH = "faiss_index"

# def setup_llm_and_retriever():
#     """
#     Loads the LLM and the pre-built FAISS index.
#     """
#     global retriever, llm

#     # --- Setup LLM ---
#     print("Loading the LLM...")
#     # This now happens only once
#     # llm = ChatOllama(model="phi3:mini", temperature=0.2)
#     llm = ChatGroq(model='openai/gpt-oss-120b')
#     # llm = load_finetuned_llm()
#     # tokenizer = AutoTokenizer.from_pretrained("Cydonia01/llama2-medical-finetuned")
#     # llm = AutoModelForCausalLM.from_pretrained("Cydonia01/llama2-medical-finetuned")
#     print("LLM loaded.")

#     # --- Setup Retriever ---
#     if not os.path.exists(VECTOR_STORE_PATH):
#         print(f"Error: Vector store not found at '{VECTOR_STORE_PATH}'.")
#         print("Please run 'create_vector_store.py' first to create it.")
#         return

#     print("Loading the vector store...")
#     embeddings = embedding_creation()
    
#     # Load the local FAISS index
#     vector_store = FAISS.load_local(
#         VECTOR_STORE_PATH, 
#         embeddings,
#         allow_dangerous_deserialization=True 
#     )
    
#     # CODE for RERANKING The raw retrieved context
#     base_retriever = vector_store.as_retriever(search_kwargs={'k': 20})

#     # initializing the flashrank to rerank these 20 chunks and provide the top 5
#     compressor = FlashrankRerank(top_n=5)

#     # wrap them together
#     retriever = ContextualCompressionRetriever(
#         base_compressor=compressor,
#         base_retriever=base_retriever
#     )

#     # retriever = vector_store.as_retriever(search_kwargs={'k': 4})
#     print("Contextual Compression Retriever (Reranker) is ready.")


# --- Flask Routes ---

@app.route("/")
def landing():
    """Renders the landing page."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('landing.html')

@app.route("/chat")
@login_required
def index():
    """Renders the main chat page."""
    return render_template('index.html')

@app.route("/signup", methods=["POST"])
def signup():
    """Handles user registration."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "")

    if not username or not email or not password:
        return jsonify({"error": "All fields are required"}), 400

    # Check if user already exists
    if User.query.filter((User.username == username) | (User.email == email)).first():
        return jsonify({"error": "Username or Email already registered"}), 400

    # Create new user with hashed password
    hashed_password = generate_password_hash(password, method="scrypt")
    new_user = User(username=username, email=email, password_hash=hashed_password)
    
    try:
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return jsonify({"success": True, "message": "Account created successfully"}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route("/login", methods=["POST"])
def login():
    """Handles user login."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No credentials provided"}), 400

    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    user = User.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({"error": "Invalid username or password"}), 401

    login_user(user)
    return jsonify({"success": True, "message": "Logged in successfully"}), 200

@app.route("/logout")
@login_required
def logout():
    """Handles user logout."""
    logout_user()
    return jsonify({"success": True, "message": "Logged out successfully"}), 200

@app.route("/get_response", methods=["POST"])
@login_required
def get_response():
    # """Handles the user's question and returns the chatbot's response."""
    # if not retriever or not llm:
    #     return jsonify({"error": "Chatbot is not set up correctly. Please check server logs."})

    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"error": "No question provided."})

    # # Retrieve relevant documents
    # retrieved_docs = retriever.invoke(user_question)
    # context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # # Use the imported prompt template
    # final_prompt = prompt_template.invoke({'context': context, 'question': user_question})

    # # Generate response using the globally loaded LLM
    # try:
    #     response = llm.invoke(final_prompt)
    #     # response = llm.stream(final_prompt)
    #     answer = response.content if hasattr(response, 'content') else str(response)
    # except Exception as e:
    #     answer = f"An error occured: {str(e)}"
    
    result = crag_app.invoke({
    "question": user_question,
    "docs": [], "good_docs": [], "verdict": "",
    "reason": "", "strips": [], "kept_strips": [],
    "refined_context": "", "web_query": "", "web_docs": [], "answer": ""
    })
    answer = result["answer"]

    return jsonify({"answer": answer})

# --- Main Execution ---

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
