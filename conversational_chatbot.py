# Import necessary libraries
import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage, Document
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, JSONLoader, CSVLoader
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings  # Using OpenAI embeddings for vectorization
import pinecone  # Pinecone client for vector storage
import os
from dotenv import load_dotenv  # Loading environment variables
import pandas as pd
from langchain.schema import Document


# Load environment variables from a .env file (GROQ API key, OpenAI API key, Pinecone API key, and environment)
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Replace HuggingFace embeddings with OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
# Importing Pinecone-specific classes to set up connection and configuration
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone API key and environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")

# Initializing Pinecone client with API key and specifying environment via ServerlessSpec
# Initialize Pinecone with the correct API key and environment


# Initialize Pinecone client with the specified API key and environment
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
# Define or create an index if it doesn't already exist
index_name = "rcm-new-applications"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1500,  # Embedding dimension must match the OpenAI embedding model dimension (1536 for text-embedding-ada-002)
        metric="cosine"  # Using cosine similarity as the distance metric
    )

# Retrieve the index for further operations
index = pc.Index(index_name)  # Retrieve or interact with the specified index

# Set up Streamlit user interface with title and introductory text
st.set_page_config(page_title="Onasi RCM Chatbot")
st.title("Onasi RCM Chatbot")
st.write('Welcome, happy to ask any questions you may have!')

# Initialize ChatGroq model using the provided API key
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Document loading and processing (only if 'final_documents' is not already stored in session state)
if 'final_documents' not in st.session_state:
    # Load PDF file using PyPDFLoader and store it with metadata for identifying document type
    st.session_state.loader_pdf = PyPDFLoader("./Onasi_RCM.pdf")
    st.session_state.docs_pdf = [
        Document(page_content=doc.page_content, metadata={"file_type": "pdf", "source": "Onasi_RCM.pdf"})
        for doc in st.session_state.loader_pdf.load()
    ]
    
    

    # Load JSON data file and batch-process it as needed
    jq_schema = '.[]'
    st.session_state.loader_json = JSONLoader("./Dataset.json", jq_schema=jq_schema, text_content=False)
    st.session_state.loader_json_excel = JSONLoader("./excel-to-json.json", jq_schema=jq_schema, text_content=False)
        
    # Define file paths and batch size
    file_paths = ["./Dataset.json", "./excel-to-json.json"]  # Combine file paths into a single list
    batch_size = 1000

    # Initialize list for Document objects
    combined_json_documents = []

    # Process each JSON file and append to a single list
    for file_path in file_paths:
        data = pd.read_json(file_path)
        for _, row in data.iterrows():
            combined_json_documents.append(
                Document(
                    page_content=str(row.to_dict()),
                    metadata={"file_type": "json", "source": file_path.split('/')[-1]}  # Add source dynamically
                )
            )

    # Combine PDF and JSON documents
    st.session_state.docs = st.session_state.docs_pdf + combined_json_documents
                
    
    # Split documents into chunks for embedding, using specified chunk size and overlap
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

    # Initialize OpenAI embeddings
    # st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.pinecone_index = Pinecone(index_name=index_name)

    # Prepare documents for insertion into Pinecone
    # Prepare documents for insertion into Pinecone
    docs_to_index = []
    for doc in st.session_state.final_documents:
        embedding = embeddings.embed_query(doc.page_content)
        docs_to_index.append({
            "id": doc.metadata.get("id", str(hash(doc.page_content))),
            "values": embedding,
            "metadata": {"page_content": doc.page_content, **doc.metadata}  # Include page_content in metadata
        })

    # Batch upsert (in case of a large number of vectors, batch them into smaller chunks)
    batch_size = 500  # Choose an appropriate batch size
    for i in range(0, len(docs_to_index), batch_size):
        batch = docs_to_index[i:i + batch_size]
        index.upsert(vectors=batch)  # Use the index object for upserting

# Function to retrieve relevant chunks of documents based on a user query, with an optional filter for document type
def retrieve_relevant_chunks(question, num_chunks=3, file_type=None):
    # Generate an embedding (vector representation) of the user query
    question_embedding = embeddings.embed_query(question)
    
    # Create a filter dictionary for the query based on the specified file type, if provided
    filter_dict = {"file_type": file_type} if file_type else {}
    
    # Query the Pinecone index to retrieve the most similar documents
    similar_docs = index.query(
        vector=question_embedding,         # Use the query embedding as the search vector
        top_k=num_chunks,                  # Retrieve the top 'num_chunks' most similar documents
        filter=filter_dict,                # Apply the filter if a file type is specified, or use no filter
        include_values=True,               # Include the actual vector values of the matching documents in the response
        include_metadata=True              # Include the metadata (e.g., file type, source) of the matching documents in the response
    )
    
    # Extract and concatenate the content of the matching documents
    # If 'page_content' is missing in the metadata, return an empty string for that match
    return "\n".join([match["metadata"].get("page_content", "") for match in similar_docs["matches"]])



# Define the template for generating prompts with context and input placeholders
prompt_template = ChatPromptTemplate.from_template(
    """
    You are a specialized conversational chatbot designed to accurately retrieve and provide information on medical codes, Validation rules,
    and the RCM (Revenue Cycle Management) application. Only answer from the provided context. So we have RCM manual, Nphies Validation and medical coding data
    You are an expert in these three, the first source of information should be these files, if you cannot find information from them, you will give an informative answer about 
    what could be a potential answer or where can you find such information. 
    
    Focus on Rule IDs and their descriptions. If a specific Rule ID is mentioned (e.g., BV-00012), ensure it is retrieved from the provided JSON data.

    
    Use the JSON data as your primary reference to answer questions 
    about specific codes, their values, and descriptions. Also search in Onasi RCM if the user asks for RCM application.
    
    Answer accurately and succintly to questions based on the provided context. 
   
    Do not say from this file, always respond with within my information or scope ...

    Instructions:
    - You are a friendly chatbot who welcomes the user to ask any questions and happy to help any time, do not say context is quite expensive.
    - If there are many codes the user is asking about, provide information in table format.
    - Search meticuously for all validation rules like BV-0002 and provide relevant information when asked from the Nphies Validation file.
    - If the user asks for a specific code, provide the code and its description.
    - Give the most pertinant answer possible, if information is too much, output in table format or just a summary.
    - Do not give answers more than 4 lines.

    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Function to generate a response based on the user query and context
def get_chatgroq_response(question, file_type=None):
    # Retrieve relevant context chunks based on question and optional file_type filter
    context = retrieve_relevant_chunks(question, file_type=file_type)
    
    # Format the prompt with context and question
    formatted_prompt = prompt_template.format(context=context, input=question)
    
    # Create a conversational flow with message history for context
    flow_messages = [SystemMessage(content="You are a conversational AI assistant.")]
    for entry in st.session_state['chat_history']:
        flow_messages.append(HumanMessage(content=entry['question']))
        flow_messages.append(AIMessage(content=entry['answer']))
    
    # Add the formatted prompt as the user's current question
    flow_messages.append(HumanMessage(content=formatted_prompt))
    
    # Get the response from the ChatGroq model
    answer = llm(flow_messages)
    
    return answer.content

# Display previous chat history (maintaining chat session in Streamlit)
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

for entry in st.session_state['chat_history']:
    st.markdown(f"**You:** {entry['question']}")
    st.markdown(f"**AI:** {entry['answer']}")

# Use a form to handle user input for the chatbot
with st.form(key='chat_form', clear_on_submit=True):
    model_input = st.text_input("Type your question here:", key=f"input_{len(st.session_state['chat_history'])}", label_visibility="collapsed")
    submit_button = st.form_submit_button("Ask")

# Process and store user input when the form is submitted
if submit_button and model_input:
    with st.spinner('Processing...'):
        response = get_chatgroq_response(model_input)
        
        # Append question-response pair to chat history
        st.session_state['chat_history'].append({"question": model_input, "answer": response})
        
        # Rerun Streamlit to display updated chat history
        st.rerun()

# Add a reset button to clear chat history
if st.button("Reset Chat"):
    st.session_state.clear()  # Clear all session state variables on reset
