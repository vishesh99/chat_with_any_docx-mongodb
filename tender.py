import streamlit as st
import os
import re
import pdfplumber
import pandas as pd
from PyPDF2 import PdfReader
import asyncio
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check if environment variables are loaded correctly
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set correctly")

# Configure Google Generative AI
genai.configure(api_key=google_api_key)

# Define the directory to save indices
INDEX_DIR = "indices"
if not os.path.exists(INDEX_DIR):
    os.makedirs(INDEX_DIR)

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

async def get_vector_store(text_chunks, index_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(os.path.join(INDEX_DIR, index_name))

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "answer is not available in the context".
    you can use online maps for the answer,you can use maps to answer question related to distance 
     Don't provide the wrong answer.the document which i am passing to you is 
    a legal government document , so i need the exact answers from the following documents and the answer should not be changed in any manner.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

async def user_input(user_question, index_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store_path = os.path.join(INDEX_DIR, index_name)
    new_db = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    # Remove all non-English text
    english_text = re.sub(r'[^a-zA-Z0-9\s,.!?\'"()-]', '', text)

    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', english_text).strip()

    return cleaned_text

def read_questions_from_file(questions_file):
    questions = questions_file.read().splitlines()
    return questions

def write_responses_to_excel(responses, excel_path):
    df = pd.DataFrame(responses, columns=['Question', 'Answer'])
    df.to_excel(excel_path, index=False)

async def process_questions(questions, index_name):
    responses = []
    for question in questions:
        response = await user_input(question, index_name)
        responses.append({'Question': question, 'Answer': response['output_text']})
    return responses

def main():
    st.title("PDF Document Processing")

    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    uploaded_questions = st.file_uploader("Upload a text file with questions", type=["txt"])

    if uploaded_pdf is not None and uploaded_questions is not None:
        # Save uploaded files
        pdf_file_path = os.path.join("uploaded_files", uploaded_pdf.name)
        with open(pdf_file_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        questions = read_questions_from_file(uploaded_questions)

        text = extract_text_from_pdf(pdf_file_path)
        text_chunks = get_text_chunks(text)
        index_name = uploaded_pdf.name
        asyncio.run(get_vector_store(text_chunks, index_name))
        st.success(f"File '{uploaded_pdf.name}' processed and indexed as '{index_name}'")

        st.header("Ask Questions:")
        if st.button("Answer Questions"):
            responses = asyncio.run(process_questions(questions, index_name))
            st.success("Questions answered.")

            # Save responses to Excel
            excel_file_name = os.path.splitext(uploaded_pdf.name)[0] + ".xlsx"
            excel_file_path = os.path.join("responses", excel_file_name)
            if not os.path.exists("responses"):
                os.makedirs("responses")
            write_responses_to_excel(responses, excel_file_path)
            st.success(f"Responses saved to '{excel_file_path}'.")

if __name__ == "__main__":
    main()
