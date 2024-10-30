import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables and API keys
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
# os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="Onasi RCM Chatbot")
st.title("Onasi RCM Chatbot")
st.write('Welcome, happy to ask any questions you may have!')

# Initialize Groq LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Load and split PDF document into chunks if not already done
if 'final_documents' not in st.session_state:
    st.session_state.loader = PyPDFLoader("./Onasi_RCM.pdf")
    st.session_state.docs = st.session_state.loader.load()
    
    # Use RecursiveCharacterTextSplitter for chunking and FAISS for vector storage
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Function to retrieve relevant context chunks
def retrieve_relevant_chunks(question, num_chunks=5):
    # Perform similarity search to get the most relevant chunks for the question
    similar_docs = st.session_state.vectors.similarity_search(question, k=num_chunks)
    return "\n".join([doc.page_content for doc in similar_docs])

# Define the prompt template with placeholders for context and input
prompt_template = ChatPromptTemplate.from_template(
    """
    You are a helpful conversational chatbot. Answer the questions based on the provided context and the previous conversation.
    Please provide the most accurate response. You will first understand what the user is asking, and reply based on that accurately from the context and if not 
    then use common sense.
    
    Like at the start, you need to gather more information from the user, and if the user does not ask a question, then tell him or her to, dont give too long responses
    
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Function to get the response from the Groq model
def get_chatgroq_response(question):
    # Retrieve relevant context chunks
    context = retrieve_relevant_chunks(question)
    
    # Format the prompt with the context and question
    formatted_prompt = prompt_template.format(context=context, input=question)
    
    # Create flow messages, including previous conversation history
    flow_messages = [SystemMessage(content="You are a conversational AI assistant.")]
    for entry in st.session_state['chat_history']:
        flow_messages.append(HumanMessage(content=entry['question']))
        flow_messages.append(AIMessage(content=entry['answer']))
    
    # Add the current question to the flow
    flow_messages.append(HumanMessage(content=formatted_prompt))
    
    # Get the response from the LLM
    answer = llm(flow_messages)
    
    return answer.content

# Display previous chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

for entry in st.session_state['chat_history']:
    st.markdown(f"**You:** {entry['question']}")
    st.markdown(f"**AI:** {entry['answer']}")

# Use a form to handle text input and pressing enter
with st.form(key='chat_form', clear_on_submit=True):
    model_input = st.text_input("Type your question here:", key=f"input_{len(st.session_state['chat_history'])}", label_visibility="collapsed")
    submit_button = st.form_submit_button("Ask")

# If the form is submitted
if submit_button and model_input:
    with st.spinner('Processing...'):
        # Get response from the Groq model with relevant context
        response = get_chatgroq_response(model_input)
        
        # Append the current question-response pair to chat history
        st.session_state['chat_history'].append({"question": model_input, "answer": response})
        
        # Display the updated chat history
        st.rerun()

# Add a reset button to clear chat history
if st.button("Reset Chat"):
    st.session_state.clear()  # Clears all session state variables on reset
