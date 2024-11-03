# app.py
from flask import Flask, render_template, request, jsonify
from langchain.llms import OpenAI  # Import from langchain instead of langchain_openai
from langchain.prompts import PromptTemplate
import os
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Check if API key is loaded
if not api_key:
    raise ValueError("API key not found. Make sure it's set in the .env file.")

# Read the prompt from the specified file
try:
    with open('pdf_text', 'r') as file:
        prompt = file.read()
except FileNotFoundError:
    raise FileNotFoundError("The file 'pdf_text' was not found.")

# Set up the assistant template
hotel_assistant_template = prompt + """
You are "Sai Mahadasa", a professional assistant specialized in providing brief, clear responses to questions about my career background. 

- Only respond to questions directly related to my professional experience, contact information, skills, and projects. 
- Keep answers short and conversational, focusing on providing relevant insights about my background.
- If a question falls outside my professional scope, respond with, "I'm unable to assist with that, sorry!"

Question: {question} 
Answer:
"""

# Initialize the PromptTemplate
hotel_assistant_prompt_template = PromptTemplate(
    input_variables=["question"],
    template=hotel_assistant_template
)

# Initialize OpenAI model
llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0, openai_api_key=api_key)
llm_chain = hotel_assistant_prompt_template | llm

# Function to query the LLM (Language Model)
def query_llm(question):
    response = llm_chain.invoke({'question': question})
    return response

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"success": False, "message": "Question is required."}), 400

    response = query_llm(question)
    
    return jsonify({
        "success": True,
        "response": response
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get the PORT from the environment or default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)

