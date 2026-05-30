import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("⚠️ Google API Key not found! Please add GOOGLE_API_KEY to your .env file")


#####################################################
# FUNCTION TO READ THE TEXT FROM PDFs
#####################################################
def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


#####################################################
# FUNCTION TO DIVIDE THE TEXT FROM PDFs INTO SMALLER CHUNKS
#####################################################
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


#####################################################
# GET TEXT EMBEDDINGS USING GEMINI MODEL & PREPARING THE VECTOR DATABASE
#####################################################
def get_embeddings(chunks):
    api_key = os.getenv('GOOGLE_API_KEY')
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectors = FAISS.from_texts(chunks, embedding=embeddings)
    vectors.save_local("faiss_index")


#####################################################
# PREPARING THE CONVERSATIONAL CHAIN
#####################################################
def get_conversational_chain():
    prompt_temp = '''
    Answer the question from the provided context. Try to answer in as detailed manner as possible from the provided context.
    If the answer to the question is not known from the provided context, then dont provide wrong answers, in that case just say,
    'Answer to the question is not available in the provided document. Feel free to ask question from the provided context.'
    Context:\n{context}\n
    Question:\n{question}\n
    '''
    prompt = PromptTemplate(
        template=prompt_temp,
        input_variables=['context', 'question']
    )

    api_key = os.getenv('GOOGLE_API_KEY')
    model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.5, google_api_key=api_key)
    chain = prompt | model
    return chain


#####################################################
# PREPARING THE MODEL'S RESPONSE
#####################################################
def get_response(user_input):
    # Check if FAISS index exists
    if not os.path.exists('faiss_index'):
        st.error("❌ No PDFs processed yet! Please upload and process PDFs first using the sidebar.")
        return
    
    api_key = os.getenv('GOOGLE_API_KEY')
    embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=api_key)
    new_db = FAISS.load_local('faiss_index', embedding, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_input)  # conversion of 'user_input' into embeddings happens implicitly within the 'similarity_search' method
    
    # Format context from documents
    context = "\n".join([doc.page_content for doc in docs])
    
    chain = get_conversational_chain()
    response = chain.invoke({
        'context': context,
        'question': user_input
    })
    
    # Extract text from response
    output_text = response.content if hasattr(response, 'content') else str(response)
    print(output_text)
    st.write('Reply: ', output_text)


#####################################################
# CREATING THE FRONT END APPLICATION
#####################################################
def main():
    st.title('Chat With PDF Using Gemini 🤖')
    user_question = st.text_input(
        label='Ask a question related to the uploaded PDFs.',
        label_visibility='hidden',
        placeholder='Type your question here...'
    )
    if user_question and len(user_question) > 0:
        get_response(user_question)
    with st.sidebar:
        st.title('Upload PDF 📄')
        documents = st.file_uploader(
            label='Upload PDFs',
            label_visibility='hidden',
            type=['pdf'],
            accept_multiple_files=True
        )
        if st.button('Submit and Process'):
            with st.spinner('In Process...'):
                text = get_pdf_text(documents)
                chunks = get_text_chunks(text)
                get_embeddings(chunks)
                st.success('DONE!')


if __name__ == '__main__':
    main()
