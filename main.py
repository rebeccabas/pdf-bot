import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging

class PDFChatbot:
    def __init__(self, db_path):
        self.db_path = db_path
        self.qa_chain = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def setup_chat(self):
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = FAISS.load_local(
                self.db_path, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            llm = Ollama(
                model="llama2:7b",
                temperature=0.7,
                base_url="http://localhost:11434"
            )
            
            # Define prompt template to strictly use vector DB content
            prompt_template = """You are a helpful assistant that provides answers based solely on the content from the vector database. 

            Based only on the above excerpts, please answer the following question. If the answer is not contained in the excerpts, say "I cannot find this information in the database.":

            Question: {question}
            Answer: """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={
                    "prompt": PROMPT
                }
            )
            
            self.logger.info("Chatbot setup completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error setting up chatbot: {str(e)}")
            st.error(f"Error setting up chatbot: {str(e)}")
            return False
    
    def ask_question(self, question):
        if not self.qa_chain:
            raise ValueError("Please initialize chatbot first by calling setup_chat()")
        try:
            return self.qa_chain.run(question)
        except Exception as e:
            self.logger.error(f"Error processing question: {str(e)}")
            st.error(f"Error processing question: {str(e)}")
            return None

def main():
    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    # Custom CSS for chat interface
    st.markdown("""
        <style>
        .stTextInput {
            position: fixed;
            bottom: 3rem;
            background-color: white;
            padding: 1rem;
            z-index: 100;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .chat-message.user {
            background-color: #e6f3ff;
            border: 1px solid #b3d9ff;
        }
        .chat-message.assistant {
            background-color: #f0f0f0;
            border: 1px solid #ddd;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    st.title("📚 PDF Chatbot")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize chatbot
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = PDFChatbot("faiss_db")
        setup_success = st.session_state.chatbot.setup_chat()
        if setup_success:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Hello! I'm ready to help you with questions about your PDF documents. I'll answer based only on the information in the database. What would you like to know?"
            })
        else:
            st.error("Failed to initialize chatbot. Please check your setup.")
            st.stop()

    # Chat container
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.container():
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message user"><b>You:</b><br>{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant"><b>Assistant:</b><br>{message["content"]}</div>', unsafe_allow_html=True)

    # Create a container for the input box
    input_container = st.container()
    
    # Input box at the bottom
    with input_container:
        col1, col2 = st.columns([6,1])
        with col1:
            question = st.text_input("", placeholder="Type your message here...")
        with col2:
            send_clicked = st.button("Send", use_container_width=True)

        if send_clicked and question:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })
            
            # Get bot response
            with st.spinner(""):
                answer = st.session_state.chatbot.ask_question(question)
                if answer:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
            
            st.rerun()

    # Add a clear chat button at the top
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()