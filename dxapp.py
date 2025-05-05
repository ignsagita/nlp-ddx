import streamlit as st
import os
import pickle
import json
import pandas as pd
import csv
import io
from datetime import datetime


from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage


# Process PDFs and generate diagnosis
def process_input(question):
    model_local = Ollama(model="mistral")

    # Load the vector database
    vectorstore = Chroma(persist_directory="./db", embedding_function=OllamaEmbeddings(model='nomic-embed-text'))
    retriever = vectorstore.as_retriever()

    after_rag_template = """ Answer the question based only on the following context,:{context} 
    Question:{question} 
    What are the top 10 most likely diagnoses? Be precise, listing one diagnosis per line, and try to cover many unique possibilities (at least 10).
    Ensure the order starts with the most likely. The top 10 diagnoses are."""
    # print(after_rag_template)

    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    # print(after_rag_prompt)
    after_rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | after_rag_prompt
            | model_local
            | StrOutputParser()
    )

    return after_rag_chain.invoke(question)

# Process CSV symptoms and generate diagnoses
def process_csv(csv_data):
    model_local = Ollama(model="mistral")
    
    # Load the vector database
    vectorstore = Chroma(persist_directory="./db", embedding_function=OllamaEmbeddings(model='nomic-embed-text'))
    retriever = vectorstore.as_retriever()
    
    results = []
    
    # For each row in the CSV (each symptom set)
    for index, row in csv_data.iterrows():
        # Convert the row to a case description
        symptoms = ', '.join([f"{col}: {val}" for col, val in row.items() if pd.notna(val) and val != ""])
        case_description = f"Patient presents with the following symptoms: {symptoms}"
        
        # Use the RAG to get diagnoses
        after_rag_template = """Answer the question based only on the following context: {context} 
        
        Question: Given the following patient symptoms, what are the top 10 most likely diagnoses? {question}
        
        List only the diagnoses, one per line, with no additional text. Start with the most likely."""

        after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
        after_rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | after_rag_prompt
                | model_local
                | StrOutputParser()
        )
        
        diagnoses = after_rag_chain.invoke(case_description)
        
        # Process the diagnoses
        diagnosis_list = [d.strip() for d in diagnoses.strip().split('\n') if d.strip()]
        
        # Add to results
        result_row = row.to_dict()
        for i, diagnosis in enumerate(diagnosis_list[:5], 1):
            result_row[f'Diagnosis_{i}'] = diagnosis
            
        results.append(result_row)
    
    # Convert results to DataFrame
    result_df = pd.DataFrame(results)
    return result_df


# streamlit UI
st.title("Differential Diagnosis with Medical Textbook RAG")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Single Case Diagnosis", "Batch CSV Processing"])

with st.sidebar:
    st.write("Upload your Medical Textbook in PDF format")
    # UI for input fields

    UploadedFiles = st.file_uploader("Upload your Medical Textbook in PDF format here and click on 'Upload'", accept_multiple_files=True)

    # persisting the Chromadb Database
    persist_directory = "./db"

    if st.button("Upload"):
        try:
            os.mkdir("UploadedTextbook")
        except:
            print("File already exists")
        with st.spinner("Processing SOPs"):
            # get the pdf text
            DocumentList = []
            for UploadedFile in UploadedFiles:
                with open(os.path.join("UploadedTextbook", UploadedFile.name), "wb") as f:
                    f.write(UploadedFile.getbuffer())
                DocumentList.append(os.path.join("UploadedTextbook", UploadedFile.name))

            docs = [PyPDFLoader(pdf).load() for pdf in DocumentList]
            docs_list = [item for sublist in docs for item in sublist]

            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
            doc_splits = text_splitter.split_documents(docs_list)

            vectorstore = Chroma.from_documents(
                documents=doc_splits,
                # collection_name="rag-chroma",
                embedding=OllamaEmbeddings(model='nomic-embed-text'),
                persist_directory=persist_directory,
            )
            vectorstore.persist()
            vectorstore = None

        st.success("Textbooks Uploaded and Processed Successfully!")

# Tab for single case diagnosis
with tab1:
    question = st.text_area("Enter the Case Description", height=150)

    if st.button('List differential diagnosis'):
        if not question:
            st.error("Please enter a case description")
        else:
            with st.spinner('Processing your case...'):
                answer = process_input(question)
                st.text_area("Differential Diagnosis", value=answer, height=300)

# Tab for CSV batch processing
with tab2:
    st.write("Upload a CSV file with patient symptoms to get diagnoses")
    
    uploaded_csv = st.file_uploader("Upload symptoms CSV file", type="csv")
    
    if uploaded_csv is not None:
        try:
            # Read the CSV
            df = pd.read_csv(uploaded_csv)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if st.button('Process CSV and Generate Diagnoses'):
                with st.spinner('Processing all cases in CSV...'):
                    # Process the CSV data
                    result_df = process_csv(df)
                    
                    # Display the results
                    st.write("Generated Diagnoses:")
                    st.dataframe(result_df)
                    
                    # Provide download link for the results
                    csv_data = result_df.to_csv(index=False)
                    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"diagnoses_results_{current_time}.csv"
                    
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing the CSV file: {str(e)}")
            st.error("Please ensure your CSV is properly formatted with symptom columns")