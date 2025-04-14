from flask import Flask, request, jsonify
import os
from openai import AzureOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
CORS(app)

# ✅ Load configs from .env
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# LangSmith (optional)
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "false")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "")

# 🔐 Safety check
if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, OPENAI_API_VERSION, DEPLOYMENT_NAME]):
    raise EnvironmentError("❌ Missing Azure OpenAI config in .env")

# Initialize Azure OpenAI Client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# Global variables
vector_store = None
loaded_text = None
feedback_log = []

# ✅ Endpoint: Load custom text
@app.route('/load-text', methods=['POST'])
def load_text():
    global vector_store, loaded_text

    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "Missing 'text' field in JSON."}), 400

    loaded_text = data['text']

    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts([loaded_text], embeddings)
        return jsonify({"message": "Text successfully loaded and indexed."}), 200
    except Exception as e:
        return jsonify({"error": f"Vector store error: {str(e)}"}), 500

# ✅ Endpoint: Ask questions
@app.route('/query', methods=['POST'])
def query_text():
    global vector_store

    if vector_store is None:
        return jsonify({"error": "No text has been loaded yet."}), 400

    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json()
    query = data.get("query", "")

    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        messages_prompt = [
            {"role": "system", "content": "You are an intelligent assistant. Use the retrieved context to answer the query."},
            {"role": "user", "content": f"Query: {query}\n\nContext: {context}"}
        ]

        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=messages_prompt,
            temperature=0.2,
            max_tokens=4096
        )

        answer = response.choices[0].message.content
        return jsonify({"query": query, "answer": answer})

    except Exception as e:
        return jsonify({"error": f"OpenAI API error: {str(e)}"}), 500

# ✅ Feedback with re-answering on dissatisfaction
@app.route('/feedback', methods=['POST'])
def save_feedback():
    global feedback_log, vector_store

    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json()
    required_fields = ["query", "answer", "feedback"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing one of: query, answer, feedback"}), 400

    query = data["query"]
    answer = data["answer"]
    feedback = data["feedback"]

    feedback_entry = {
        "query": query,
        "answer": answer,
        "feedback": feedback
    }

    feedback_log.append(feedback_entry)
    print("📥 Feedback received:", feedback_entry)

    # 🔁 Retry if user is not satisfied
    if feedback.lower() == "not_satisfied":
        try:
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            retrieved_docs = retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in retrieved_docs])

            messages_prompt = [
                {"role": "system", "content": "You are an intelligent assistant. Try to rephrase your answer more clearly and provide alternate useful information if possible."},
                {"role": "user", "content": f"Query: {query}\n\nContext: {context}"}
            ]

            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=messages_prompt,
                temperature=0.7,
                max_tokens=4096
            )

            new_answer = response.choices[0].message.content
            return jsonify({
                "message": "Alternate answer generated.",
                "query": query,
                "answer": new_answer
            })

        except Exception as e:
            return jsonify({"error": f"Retry generation failed: {str(e)}"}), 500

    # 👍 If satisfied
    return jsonify({"message": "Feedback saved. Thank you!"}), 200

# ✅ Health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running"}), 200

if __name__ == '__main__':
    app.run(debug=True)
