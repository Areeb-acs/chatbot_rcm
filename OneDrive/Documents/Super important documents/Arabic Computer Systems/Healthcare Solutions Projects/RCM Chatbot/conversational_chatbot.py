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
from PIL import Image  # For loading logo images

from dotenv import load_dotenv  # Loading environment variables
import pandas as pd
from langchain.schema import Document
from tqdm import tqdm
import multiprocessing

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
# Load environment variables from a .env file (GROQ API key, OpenAI API key, Pinecone API key, and environment)
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Replace HuggingFace embeddings with OpenAI embeddings
embeddings = OpenAIEmbeddings()
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
index_name = "rcm-app-3"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Embedding dimension must match the OpenAI embedding model dimension (1536 for text-embedding-ada-002)
        metric="euclidean"  # Using cosine similarity as the distance metric
    )

# Retrieve the index for further operations
index = pc.Index(index_name)  # Retrieve or interact with the specified index

# Set up Streamlit user interface with title and introductory text
# Set up Streamlit app configurations
st.set_page_config(
    page_title="Revenue Cycle Management Chatbot",
    page_icon="💬",  # You can replace this with a custom emoji or icon
    layout="wide"
)

# Display a logo at the top of the app
logo = Image.open("./onasi_logo.png")  # Replace with your logo's file path
# Create three columns with adjusted widths
col1, col2, col3 = st.columns([3, 3, 1])  # Make col1 wider to push the image right

# Place the image in the center-right column
with col2:
    st.image(logo, width=300)  # Adjust width as needed


# Add the language dropdown in the third column
with col3:
    language = st.selectbox("Language", options=["English", "Arabic"], index=0)
    
# App title and introductory text
st.title("RCM Chatbot")
st.write('Welcome, happy to ask any questions you may have!')

# Inject custom CSS for styling
st.markdown(
    """
    <style>
    /* Set the textbox background to white */




    </style>
    """,
    unsafe_allow_html=True
)


# Initialize ChatGroq model using the provided API key
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

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
        
    # Define file paths and batch size
    file_paths = ["./Medical_coding.json", "./Nphies_validation.json"]  # Combine file paths into a single list
    batch_size = 400

    # Initialize list for Document objects
    combined_json_documents = []

    # Process each JSON file using pandas
    for file_path in file_paths:
        try:
            data = pd.read_json(file_path)
            for _, row in data.iterrows():
                combined_json_documents.append(
                    Document(
                        page_content=str(row.to_dict()),
                        metadata={"file_type": "json", "source": file_path.split('/')[-1]}  # Add source dynamically
                    )
                )
        except ValueError as e:
            st.error(f"Error processing JSON file {file_path}: {e}")

    # Combine PDF and JSON documents
    st.session_state.docs = st.session_state.docs_pdf +  combined_json_documents

                
    
    # Split documents into chunks for embedding, using specified chunk size and overlap
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

    # Initialize OpenAI embeddings
    # st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.pinecone_index = Pinecone(index_name=index_name)

    chunks = [doc.page_content for doc in st.session_state.final_documents]
    embedding_batches = []

    # Generate embeddings using threads
    def generate_embeddings_threading(chunks, embeddings, batch_size=200):
        embeddings_list = []

        def embed_chunk(chunk):
            return embeddings.embed_query(chunk)

        with ThreadPoolExecutor() as executor:
            for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings"):
                batch = chunks[i:i + batch_size]
                embeddings_list.extend(executor.map(embed_chunk, batch))

        return embeddings_list

    # Call the function
    embedding_batches = generate_embeddings_threading(chunks, embeddings)
    # Prepare documents for insertion into Pinecone
    # Prepare documents for insertion into Pinecone
    # Prepare documents for Pinecone upsert

    docs_to_index = [
        {
            "id": doc.metadata.get("id", str(hash(doc.page_content))),
            "values": embedding_batches[idx],
            "metadata": {"page_content": doc.page_content, **doc.metadata}
        }
        for idx, doc in enumerate(st.session_state.final_documents)
    ]

    # Batch upsert (in case of a large number of vectors, batch them into smaller chunks)
    batch_size = 200  # Choose an appropriate batch size
    for i in range(0, len(docs_to_index), batch_size):
        batch = docs_to_index[i:i + batch_size]
        index.upsert(vectors=batch)  # Use the index object for upserting

# Function to retrieve relevant chunks of documents based on a user query, with an optional filter for document type
# Function to retrieve relevant chunks of documents based on a user query, with an optional filter for document type
def retrieve_relevant_chunks(question, num_chunks=8, file_type=None):
    import re
    
    # Extract numbers and Rule IDs from the query
    numbers_in_query = re.findall(r'\d+', question)  # Extract numbers
    rule_id_match = re.search(r'\bBV-\d+\b', question)  # Match Rule IDs like "BV-00027"

    # Check for exact match search based on Rule ID
    if rule_id_match:
        rule_id_to_search = rule_id_match.group(0)  # Extract matched Rule ID
        matches = [
            doc.page_content
            for doc in st.session_state.final_documents
            if rule_id_to_search in doc.page_content  # Check if the Rule ID exists in the page content
        ]
        if matches:
            return "\n".join(matches[:num_chunks])  # Return top 'num_chunks' matches

    # Check for exact match search based on numbers
    if numbers_in_query:
        number_to_search = numbers_in_query[0]  # Assume we're searching for the first number found
        matches = [
            doc.page_content
            for doc in st.session_state.final_documents
            if number_to_search in doc.page_content  # Check if the number exists in the page content
        ]
        if matches:
            return "\n".join(matches[:num_chunks])  # Return top 'num_chunks' matches

    # Fall back to embedding-based search for more complex queries
    question_embedding = embeddings.embed_query(question)
    filter_dict = {"file_type": file_type} if file_type else {}
    similar_docs = index.query(
        vector=question_embedding,
        top_k=num_chunks,
        filter=filter_dict,
        include_values=True,
        include_metadata=True,
    )
    return "\n".join([match["metadata"].get("page_content", "") for match in similar_docs["matches"]])




# Define the template for generating prompts with context and input placeholders
prompt_template = ChatPromptTemplate.from_template( 
    """
    You are a friendly conversational chatbot that remembers context across a conversation. Use the provided conversation history to understand the user's question and provide clear, concise, and accurate responses for doctors.

    Instructions:
    1. Always refer to the conversation history for context and maintain continuity in your responses.
    2. Answer questions in plain English and ensure your response is easy to understand for a doctor.
    3. When asked to summarize, base the summary only on the relevant details from the conversation history. Ignore any newly retrieved chunks or external context for summarization tasks.
    4. For requests like "summarize the above information," focus only on the most recent exchanges in the conversation history. Extract and condense the key points into a concise response.
    5. When answering non-summarization queries, you may use the retrieved context along with the conversation history to provide accurate and complete responses.

    Conversation History:
    {context}

    Current Question:
    {input}
    """
)
# prompt_template = ChatPromptTemplate.from_template(
#     """
#     You are friendly conversational chatbot that remembers names, answer the question based on the provided context. Only search given the context, do not use any other information source.
#     Please provide the most accurate response. You will first understand what the user is asking, and reply based on that accurately from the context.
    
#     You are an expert in knowing about the RCM application, medical coding and nphies validation codes.
#     You are an expert in reading .json data, so you know how to read information from json files.
    
#     If what the user asks does not exist in knowledge base, like code values or anything, just say you do not know, do not make up things.
    
#     Intructions:
#     1. When you respond, do not show the context you are searching, just give a short to medium answer, to the point, if the user ask what is the code value, just answer with number etc.
#     2. If the answer is not in the given context, mention I cannot find it, out of my knowledge base.
#     3. If you cannot find any relevant information say for example the code value is not present, just say I cannot find it, out of my knowledge base
#     4. You can read json files easily.

#     <context>
#     {context}
#     <context>
    
#     Question: {input}
#     """
# )


def handle_query(question, file_type=None):
    # Check if the question is a summarization request
    if "summarize" in question.lower() or "summarise" in question.lower() or "above" in question.lower() or "summarize the above" in question.lower():
        # Use only the conversation history for summarization
        context = "\n".join([f"User: {entry['question']}\nAI: {entry['answer']}" for entry in st.session_state['chat_history']])
    else:
        # Use both conversation history and retrieved context for other queries
        context = retrieve_relevant_chunks(question)

    # Format the prompt
    formatted_prompt = prompt_template.format(context=context, input=question)
    return formatted_prompt

# Function to generate a response based on the user query and context
def get_chatgroq_response(question, file_type=None):
    # Retrieve relevant context chunks based on question and optional file_type filter
    context = handle_query(question, file_type=file_type)
    print("==============================================")
    print(context)
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

# Define a maximum chat history size
MAX_CHAT_HISTORY_SIZE = 20  # Adjust as needed based on token limits

# Function to trim chat history
def trim_chat_history():
    if len(st.session_state['chat_history']) > MAX_CHAT_HISTORY_SIZE:
        st.session_state['chat_history'] = st.session_state['chat_history'][-MAX_CHAT_HISTORY_SIZE:]
# Display previous chat history (maintaining chat session in Streamlit)
# Function to summarize chat history
def summarize_chat_history(chat_history):
    if len(chat_history) > MAX_CHAT_HISTORY_SIZE:
        summary_prompt = "Summarize the following conversation context:\n\n"
        chat_context = "\n".join([f"User: {entry['question']}\nAI: {entry['answer']}" for entry in chat_history[:-MAX_CHAT_HISTORY_SIZE]])
        summary = llm([SystemMessage(content=summary_prompt + chat_context)]).content
        summarized_entry = {"question": "Summarized Context", "answer": summary}
        return chat_history[-MAX_CHAT_HISTORY_SIZE:] + [summarized_entry]
    return chat_history

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
        
        # # Trim chat history to maintain size limit
        # trim_chat_history()
        
                # Summarize chat history if it exceeds the maximum size
        st.session_state['chat_history'] = summarize_chat_history(st.session_state['chat_history'])
        
        # Rerun Streamlit to display updated chat history
        st.rerun()

# Add a reset button to clear chat history
# Add a reset button to clear chat history only
if st.button("Reset Chat"):
    if 'chat_history' in st.session_state:
        st.session_state['chat_history'] = []  # Clear only the chat history
    # Force Streamlit to rerun and refresh the UI
    st.rerun()