from flask import Flask, request
import os
import json
import random
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

OPENAI_API_KEY = "ENTER YOUR KEY"

app = Flask(__name__)  # Create a Flask object.
PORT = os.environ.get("PORT")  # Get PORT setting from environment.
if not PORT:
    PORT = 8080

# Initialize the OpenAI API keys
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Default quiz settings
NUM_Q = 5
DIFF = "Difficult"
TOPICS = "Cloud Computing"

DOC_PROMPT = """
    Find important concepts and keywords regarding {topics}. 
    The content will be used to generate quizzes for a cloud computing university course.
"""

PROMPT = """
    Generate {num_q} {diff} multiple-choice quiz questions regarding {topics} for a cloud computing university course, 
    using concepts and keywords from the document as well as your own knowledge.
    Answers to the questions must be factually correct.
    DO NOT reference any specific context or document in the questions. 
    Output should be (only) an unquoted json array of objects 
    with keys 'question', 'options', and 'correct_answer'.
"""

def check(args, name, default):
    if name in args:
        return args[name]
    return default

def generate_quiz_html(questions):
    """Generate HTML for quiz questions."""
    html = "<h1>Quiz Questions</h1>"
    html += "<ol>"
    for question in questions:
        html += "<li>"
        html += f"<p><strong>{question['question']}</strong></p>"
        html += "<ul>"
        for option in question['options']:
            html += f"<li>{option}</li>"
        html += "</ul>"
        html += f"<p><strong>Correct Answer:</strong> {question['correct_answer']}</p>"
        html += "</li>"
    html += "</ol>"
    return html

@app.route("/", methods=["GET"])
def generate_quiz():
    args = request.args.to_dict()
    num_q = check(args, "num_q", NUM_Q)
    diff = check(args,"diff", DIFF)
    topics = check(args, "topics", TOPICS)
    
    # Load and split the PDF document
    loader = PyPDFLoader("LecVirtualization.pdf")
    documents = loader.load_and_split()

    # Vectorize the selected document parts
    vector = FAISS.from_documents(documents, embeddings)    

    # Setting up prompt template to query from vector database
    prompt = ChatPromptTemplate.from_template("""Perform the following task using the provided context:
    <context>
    {context}
    </context>
    Task: {input}""")

    # Create a chain of processing steps for documents
    document_chain = create_stuff_documents_chain(llm, prompt)

    doc_input = DOC_PROMPT  # Prompt for processing the document

    # Process the document using the document chain
    document_chain.invoke({
        "input": doc_input,
        "context": documents 
    })

    # Create a retrieval chain to query the vector database
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    input = PROMPT.format(num_q=num_q, diff=diff, topics = topics) # Generate the prompt for quiz question generation

    # Invoke the retrieval chain to generate quiz questions
    response = retrieval_chain.invoke({
        "input": input
    })

    # Convert the string representation of JSON array to a Python dictionary
    response_dict = json.loads(response["answer"])
    questions = response_dict
    
    # Format the questions for display
    html = generate_quiz_html (questions)
    
    return html


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
