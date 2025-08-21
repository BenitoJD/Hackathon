import streamlit as st
import os
import base64
import io
from PIL import Image
from dotenv import load_dotenv
import tempfile
import numpy as np
import time
import shutil

# --- NEW IMPORTS FOR DOCUMENT PROCESSING ---
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
# --- END NEW IMPORTS ---

# LangChain components
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage

# Audio processing
try:
    from streamlit_mic_recorder import mic_recorder
    MIC_RECORDER_AVAILABLE = True
except ImportError:
    MIC_RECORDER_AVAILABLE = False

import speech_recognition as sr

# Whisper imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import torch
    import librosa
    HF_WHISPER_AVAILABLE = True
except ImportError:
    HF_WHISPER_AVAILABLE = False

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Multimodal AI Orchestrator",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load environment variables (for OpenAI API key)
load_dotenv()

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #4a4a4a;
    }
    .stChatMessage {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .st-emotion-cache-janbn0 { /* Targets the assistant message background */
        background-color: #f0f8ff;
    }
    .status-container {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .status-step {
        transition: all 0.3s ease-in-out;
        opacity: 0.5;
        font-weight: normal;
    }
    .status-step.active {
        opacity: 1;
        font-weight: bold;
        color: #667eea;
    }
    .whisper-status {
        background: #f0f8ff;
        border-left: 4px solid #667eea;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# --- NEW HELPER FUNCTION FOR TEXT EXTRACTION ---
def extract_text_from_file(uploaded_file):
    """Extracts text from various file formats."""
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    if file_extension == ".txt":
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")

    elif file_extension == ".pdf":
        if not PYMUPDF_AVAILABLE:
            st.error("PDF processing requires `PyMuPDF`. Please install it: `pip install PyMuPDF`")
            return None
        try:
            with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as doc:
                text = "".join(page.get_text() for page in doc)
            return text
        except Exception as e:
            st.error(f"Error reading PDF {uploaded_file.name}: {e}")
            return None

    elif file_extension == ".docx":
        if not DOCX_AVAILABLE:
            st.error("DOCX processing requires `python-docx`. Please install it: `pip install python-docx`")
            return None
        try:
            doc = docx.Document(io.BytesIO(uploaded_file.getvalue()))
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            st.error(f"Error reading DOCX {uploaded_file.name}: {e}")
            return None

    else:
        st.warning(f"Unsupported file format: {file_extension}. Skipping {uploaded_file.name}.")
        return None

# --- CORE CLASSES (ModelManager and VoiceProcessor are unchanged) ---

class ModelManager:
    """ Manages connections and interactions with different LLMs. """
    def __init__(self):
        self.llm = None
        self.vision_model = None
        self.embedding_model = None
        self.ollama_models = self._get_ollama_models()

    def _get_ollama_models(self):
        """Dynamically get available Ollama models."""
        try:
            import ollama
            models_info = ollama.list().get('models', [])
            return [model['name'] for model in models_info]
        except Exception:
            return []

    def configure_llm(self, provider, model_name, api_key=None):
        """ Configures the primary LLM for text and vision. """
        try:
            if provider == "OpenAI" and api_key:
                self.llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0.7)
                if model_name in ["gpt-4o", "gpt-4-turbo", "gpt-4-vision-preview"]:
                    self.vision_model = self.llm
                else:
                    self.vision_model = None
                return True, f"OpenAI model `{model_name}` configured."
            elif provider == "Ollama":
                self.llm = ChatOllama(model=model_name, temperature=0.7)
                # Simple check for vision capabilities in Ollama models
                if 'llava' in model_name.lower() or 'bakllava' in model_name.lower():
                    self.vision_model = self.llm
                else:
                    self.vision_model = None
                return True, f"Ollama model `{model_name}` configured."
        except Exception as e:
            return False, f"Error configuring model: {e}"
        return False, "Configuration failed."

    def configure_embedding_model(self, provider, api_key=None):
        """ Configures the model for creating text embeddings. """
        try:
            if provider == "OpenAI" and api_key:
                self.embedding_model = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
                return True, "OpenAI embedding model configured."
            elif provider == "Local (SentenceTransformers)":
                self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                return True, "Local embedding model loaded."
        except Exception as e:
            return False, f"Error configuring embedding model: {e}"
        return False, "Configuration failed."

    def generate_response(self, prompt, context, image_base64=None):
        """Generate response with proper error handling and multimodal support."""
        if not self.llm:
            return "❌ LLM not configured. Please select a model in the sidebar."
        try:
            system_msg = SystemMessage(content="You are a helpful multimodal AI assistant. Use the provided context to answer the user's query. If an image is provided, analyze it in relation to the user's query and the context.")
            text_content = f"Context from knowledge base:\n---\n{context}\n---\n\nUser query: {prompt}"
            message_content = [{"type": "text", "text": text_content}]
            target_model = self.llm
            if image_base64:
                if self.vision_model:
                    target_model = self.vision_model
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })
                else:
                    message_content[0]['text'] += "\n\n[Note: An image was provided, but the currently configured model does not support vision.]"
            human_msg = HumanMessage(content=message_content)
            messages = [system_msg, human_msg]
            response = target_model.invoke(messages)
            return response.content
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return f"❌ Error generating response: {str(e)}"

# --- KNOWLEDGE RETRIEVER (MODIFIED FOR CHROMADB) ---
class KnowledgeRetriever:
    """ Handles the creation and querying of a persistent vector-based knowledge base using ChromaDB. """
    def __init__(self, persist_directory="chroma_db"):
        self.vector_store = None
        self.persist_directory = persist_directory
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])

    def build_from_documents(self, documents, embedding_model):
        """Build or update a persistent Chroma vector store from documents."""
        if not embedding_model:
            st.error("❌ Embedding model is not configured. Cannot build knowledge base.")
            return False
        if not documents:
            st.error("❌ No documents provided.")
            return False
        try:
            with st.spinner("📚 Building knowledge base..."):
                texts = self.text_splitter.split_text("\n\n".join(documents))
                if not texts:
                    st.error("❌ No text chunks created from documents.")
                    return False

                if not self.vector_store:
                    if os.path.exists(self.persist_directory):
                        st.toast("Loading existing knowledge base from disk...")
                        self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=embedding_model)

                if self.vector_store:
                    self.vector_store.add_texts(texts)
                    st.success(f"✅ Knowledge base updated with {len(texts)} new text chunks.")
                else:
                    st.toast("Creating new knowledge base...")
                    self.vector_store = Chroma.from_texts(
                        texts=texts,
                        embedding=embedding_model,
                        persist_directory=self.persist_directory
                    )
                    st.success(f"✅ New knowledge base created with {len(texts)} text chunks.")
                return True
        except Exception as e:
            st.error(f"❌ Failed to build knowledge base: {e}")
            return False

    def retrieve(self, query, k=3):
        """Retrieve relevant documents with error handling."""
        if not self.vector_store and os.path.exists(self.persist_directory) and st.session_state.model_manager.embedding_model:
            self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=st.session_state.model_manager.embedding_model)

        if not self.vector_store:
            return []
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
            relevant_docs = retriever.invoke(query)
            return [doc.page_content for doc in relevant_docs]
        except Exception as e:
            st.warning(f"⚠️ Could not retrieve documents: {e}")
            return []

class VoiceProcessor:
    """ Enhanced voice processor with Whisper support (OpenAI API + Hugging Face fallback). """
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.openai_client = None
        self.hf_whisper_model = None
        self.hf_whisper_processor = None

    @st.cache_resource
    def _load_whisper_models(_self):
        """Load Whisper models on initialization. Use st.cache_resource to load only once."""
        hf_processor, hf_model = None, None
        if HF_WHISPER_AVAILABLE:
            try:
                hf_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
                hf_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                hf_model.to(device)
            except Exception:
                pass # Fail silently
        return hf_processor, hf_model

    def initialize_models(self):
        if not self.hf_whisper_processor and not self.hf_whisper_model:
            self.hf_whisper_processor, self.hf_whisper_model = self._load_whisper_models()

    def configure_openai_whisper(self, api_key):
        """Configure OpenAI Whisper API client."""
        if OPENAI_AVAILABLE and api_key:
            try:
                self.openai_client = openai.OpenAI(api_key=api_key)
                return True, "✅ OpenAI Whisper API configured."
            except Exception as e:
                return False, f"❌ Failed to configure OpenAI Whisper: {e}"
        return False, "❌ OpenAI library not available or no API key provided."

    def _transcribe_with_openai_whisper(self, audio_file_path):
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(model="whisper-1", file=audio_file)
            return transcript.text, None
        except Exception as e:
            return None, f"OpenAI Whisper API error: {e}"

    def _transcribe_with_hf_whisper(self, audio_file_path):
        try:
            audio_array, sampling_rate = librosa.load(audio_file_path, sr=16000)
            input_features = self.hf_whisper_processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_features
            device = next(self.hf_whisper_model.parameters()).device
            input_features = input_features.to(device)
            with torch.no_grad():
                predicted_ids = self.hf_whisper_model.generate(input_features)
            transcription = self.hf_whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription, None
        except Exception as e:
            return None, f"Hugging Face Whisper error: {e}"

    def _transcribe_with_google_sr(self, audio_bytes):
        try:
            audio_file = io.BytesIO(audio_bytes)
            with sr.AudioFile(audio_file) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio_data)
            return text, None
        except sr.UnknownValueError:
            return None, "Could not understand the audio."
        except sr.RequestError as e:
            return None, f"Google Speech Recognition service error: {e}"
        except Exception as e:
            return None, f"Google SR error: {e}"

    def process(self, audio_bytes):
        """Process audio bytes to text using Whisper (with fallbacks). Returns (text, error, method_used)."""
        if not audio_bytes:
            return None, "No audio data provided.", "None"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            temp_file_path = tmp_file.name
        try:
            if self.openai_client:
                text, error = self._transcribe_with_openai_whisper(temp_file_path)
                if text: return text, None, "OpenAI Whisper API"
            if self.hf_whisper_model and self.hf_whisper_processor:
                text, error = self._transcribe_with_hf_whisper(temp_file_path)
                if text: return text, None, "Local Whisper (HF)"
            text, error = self._transcribe_with_google_sr(audio_bytes)
            if text: return text, None, "Google Speech Recognition"
            return None, error or "All transcription methods failed.", "Failed"
        finally:
            os.unlink(temp_file_path)

    def get_status(self):
        return {
            "openai_whisper": bool(self.openai_client),
            "hf_whisper": bool(self.hf_whisper_model and self.hf_whisper_processor),
            "google_sr": True
        }

# --- SESSION STATE INITIALIZATION ---
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'model_manager': ModelManager(),
        'knowledge_retriever': KnowledgeRetriever(persist_directory="chroma_db"),
        'voice_processor': VoiceProcessor(),
        'chat_history': [],
        'uploaded_image_b64': None,
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    st.session_state.voice_processor.initialize_models()

# --- UI COMPONENTS ---

def render_sidebar():
    """ Renders the sidebar for configuration. """
    with st.sidebar:
        st.header("🛠️ Configuration")

        with st.expander("🤖 Language Model", expanded=True):
            llm_provider = st.selectbox("Select Provider", ["OpenAI", "Ollama"])
            openai_api_key = ""
            if llm_provider == "OpenAI":
                openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
                openai_model = st.selectbox("Select Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4"])
                if st.button("Configure OpenAI LLM"):
                    if not openai_api_key: st.error("Please provide an OpenAI API key.")
                    else:
                        success, msg = st.session_state.model_manager.configure_llm("OpenAI", openai_model, openai_api_key)
                        st.toast(msg, icon="✅" if success else "❌")
            elif llm_provider == "Ollama":
                if not st.session_state.model_manager.ollama_models:
                    st.warning("No Ollama models found. Is Ollama running?")
                else:
                    ollama_model = st.selectbox("Select Model", st.session_state.model_manager.ollama_models)
                    if st.button("Configure Ollama LLM"):
                        success, msg = st.session_state.model_manager.configure_llm("Ollama", ollama_model)
                        st.toast(msg, icon="✅" if success else "❌")

        with st.expander("🧩 Embedding Model"):
            embedding_provider = st.selectbox("Select Provider", ["Local (SentenceTransformers)", "OpenAI"])
            if st.button("Configure Embedding Model"):
                api_key_to_use = openai_api_key if embedding_provider == "OpenAI" and openai_api_key else os.getenv("OPENAI_API_KEY", "")
                success, msg = st.session_state.model_manager.configure_embedding_model(embedding_provider, api_key_to_use)
                st.toast(msg, icon="✅" if success else "❌")

        with st.expander("🎤 Speech Recognition"):
            if st.button("Configure OpenAI Whisper"):
                api_key_to_use = openai_api_key or os.getenv("OPENAI_API_KEY", "")
                if api_key_to_use:
                    success, msg = st.session_state.voice_processor.configure_openai_whisper(api_key_to_use)
                    st.toast(msg, icon="✅" if success else "❌")
                else: st.error("Provide OpenAI API key in the LLM section first.")
            status = st.session_state.voice_processor.get_status()
            st.markdown(f"""
            <div class="whisper-status">
                **Transcription Status:**<br>
                🔊 OpenAI Whisper: {'🟢' if status['openai_whisper'] else '🔴'}<br>
                🏠 Local Whisper: {'🟢' if status['hf_whisper'] else '🔴'}<br>
                🌐 Google SR: {'🟢' if status['google_sr'] else '🔴'}
            </div>
            """, unsafe_allow_html=True)

        # --- MODIFIED DATA SOURCES SECTION ---
        with st.expander("📚 Data Sources", expanded=True):
            st.markdown("##### From Documents")
            doc_types = ["txt"]
            if PYMUPDF_AVAILABLE: doc_types.append("pdf")
            if DOCX_AVAILABLE: doc_types.append("docx")

            uploaded_files = st.file_uploader(
                f"Upload Documents ({', '.join(doc_types)})",
                type=doc_types,
                accept_multiple_files=True,
                key="doc_uploader"
            )
            if st.button("🔨 Build KB from Documents"):
                if uploaded_files:
                    if not st.session_state.model_manager.embedding_model:
                        st.error("Configure an embedding model first.")
                    else:
                        docs_to_process = []
                        with st.spinner("Processing documents..."):
                            for file in uploaded_files:
                                st.write(f"Extracting text from `{file.name}`...")
                                text = extract_text_from_file(file)
                                if text:
                                    docs_to_process.append(text)

                        if docs_to_process:
                            st.session_state.knowledge_retriever.build_from_documents(
                                docs_to_process, st.session_state.model_manager.embedding_model
                            )
                        else:
                            st.warning("Could not extract any text from the provided documents.")
                else:
                    st.warning("Please upload document files first.")

            st.markdown("---")
            st.markdown("##### From Audio Files")
            uploaded_audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "ogg"], key="audio_uploader")
            if st.button("🔨 Build KB from Audio"):
                if uploaded_audio_file:
                    if not st.session_state.model_manager.embedding_model:
                        st.error("Configure an embedding model first.")
                    else:
                        with st.spinner("Transcribing audio... this may take a moment."):
                            audio_bytes = uploaded_audio_file.getvalue()
                            text, error, method = st.session_state.voice_processor.process(audio_bytes)
                        if error:
                            st.error(f"Audio processing failed: {error}")
                        else:
                            st.success(f"Successfully transcribed audio using {method}.")
                            st.session_state.knowledge_retriever.build_from_documents([text], st.session_state.model_manager.embedding_model)
                else:
                    st.warning("Please upload an audio file first.")
            # --- END MODIFIED SECTION ---

            kb_dir = st.session_state.knowledge_retriever.persist_directory
            if (st.session_state.knowledge_retriever.vector_store or os.path.exists(kb_dir)) and st.button("🗑️ Clear Knowledge Base"):
                try:
                    st.session_state.knowledge_retriever.vector_store = None
                    if os.path.exists(kb_dir):
                        shutil.rmtree(kb_dir)
                    st.toast("✅ Knowledge base cleared successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing knowledge base: {e}")


        st.markdown("---")
        st.subheader("📊 System Status")
        st.markdown(f"**LLM:** {'🟢 Ready' if st.session_state.model_manager.llm else '🔴 Not Configured'}")
        st.markdown(f"**Vision:** {'🟢 Ready' if st.session_state.model_manager.vision_model else '⚪ Not Available'}")
        st.markdown(f"**Embeddings:** {'🟢 Ready' if st.session_state.model_manager.embedding_model else '🔴 Not Configured'}")
        
        kb_exists = st.session_state.knowledge_retriever.vector_store or os.path.exists(st.session_state.knowledge_retriever.persist_directory)
        st.markdown(f"**Knowledge Base:** {'🟢 Ready' if kb_exists else '⚪ Empty'}")


        st.markdown("---")
        if st.button("🗑️ Clear Conversation"):
            st.session_state.chat_history = []
            st.session_state.uploaded_image_b64 = None
            st.rerun()

def render_dynamic_pipeline_status(status_placeholder, active_steps):
    """Updates the pipeline status display dynamically."""
    steps = ["Retrieval", "Vision", "Reasoning"]
    status_html = '<div class="status-container"><b>AI Pipeline Status:</b> '
    status_items = []
    for step in steps:
        active_class = "active" if step in active_steps else ""
        status_items.append(f'<span class="status-step {active_class}"> → {step}</span>')
    status_html += "".join(status_items) + "</div>"
    status_placeholder.markdown(status_html, unsafe_allow_html=True)

def handle_chat_submission(prompt):
    """Encapsulates the logic for handling a new chat prompt."""
    if not prompt:
        return
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        if st.session_state.uploaded_image_b64:
            st.image(Image.open(io.BytesIO(base64.b64decode(st.session_state.uploaded_image_b64))), width=150)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        active_steps = []
        
        render_dynamic_pipeline_status(status_placeholder, active_steps)
        time.sleep(0.3)
        retrieved_docs = st.session_state.knowledge_retriever.retrieve(prompt)
        if retrieved_docs:
            active_steps.append("Retrieval")
            render_dynamic_pipeline_status(status_placeholder, active_steps)
            with st.expander("📖 Context Retrieved from Knowledge Base"):
                st.info("\n\n".join(f"• {doc[:300]}..." for doc in retrieved_docs))
        
        time.sleep(0.3)
        if st.session_state.uploaded_image_b64:
            active_steps.append("Vision")
            render_dynamic_pipeline_status(status_placeholder, active_steps)

        time.sleep(0.3)
        active_steps.append("Reasoning")
        render_dynamic_pipeline_status(status_placeholder, active_steps)
        
        final_context = "\n\n".join(retrieved_docs) if retrieved_docs else "No relevant context found."
        response = st.session_state.model_manager.generate_response(
            prompt, final_context, st.session_state.uploaded_image_b64
        )
        status_placeholder.empty()
        st.markdown(response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        if st.session_state.uploaded_image_b64:
            st.session_state.uploaded_image_b64 = None
            st.toast("🖼️ Image context cleared after use.")

def render_chat_history():
    """Renders the main chat history."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def render_input_bar():
    """Renders the multimodal input bar at the bottom of the page."""
    uploaded_file = st.file_uploader(
        "Upload an image to chat about", type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        try:
            image_bytes = uploaded_file.getvalue()
            st.session_state.uploaded_image_b64 = base64.b64encode(image_bytes).decode()
            st.toast("✅ Image loaded and will be sent with your next message.")
            st.image(image_bytes, caption="Image ready", width=100)
        except Exception as e:
            st.error(f"Error processing image: {e}")

    if MIC_RECORDER_AVAILABLE:
        audio = mic_recorder(
            start_prompt="🎤",
            stop_prompt="⏹️",
            key='recorder',
            format="wav"
        )
        
        if audio and audio.get('bytes'):
            st.toast("🎤 Recording received, transcribing...")
            text, error, method = st.session_state.voice_processor.process(audio['bytes'])
            if error:
                st.error(f"Transcription failed: {error}")
            elif text:
                st.toast(f"Transcribed via {method}: \"{text}\"")
                handle_chat_submission(text) 

    prompt = st.chat_input("Ask a question...")
    if prompt:
        handle_chat_submission(prompt)

# --- MAIN APPLICATION LOGIC ---
def main():
    st.markdown('<h1 class="main-header">🧠 Multimodal AI Orchestrator</h1>', unsafe_allow_html=True)
    
    init_session_state()
    render_sidebar()
    
    render_chat_history()
    render_input_bar()

if __name__ == "__main__":
    main()