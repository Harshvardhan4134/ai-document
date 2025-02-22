import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def generate_embeddings(text):
    if not text or text.strip() == "":
        print("❌ Error: Empty text received for embedding generation.")
        return None

    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"❌ OpenAI Error generating embeddings: {e}")
        return None


def generate_summary(text):
    """Generates a concise summary using GPT-4."""
    if not text or text.strip() == "":
        return "No content available for summarization."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # ✅ Changed to GPT-4 instead of fine-tuned model
            messages=[
                {"role": "system", "content": "You are an AI assistant that summarizes company documents."},
                {"role": "user", "content": f"Summarize the following document and explain its key points:\n\n{text[:4000]}"}  # Limit input to 4000 characters
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        return "Summary generation failed."

def answer_question(question, context):
    """Answers a user question based on provided document context."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": 
                 "You are a company AI assistant. Answer questions ONLY based on the provided documents. "
                 "If the answer is not in the documents, say 'I do not have enough information.'"},
                {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {question}\n\nAnswer concisely."}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"❌ Error answering question: {e}")
        return "Failed to generate an answer."
