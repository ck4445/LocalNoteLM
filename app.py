import os
import sys
import subprocess
import shutil
import faiss
import numpy as np
import ollama
import PyPDF2
import json
from flask import Flask, request, jsonify, Response

# --- Ensure Ollama is installed ---
if shutil.which("ollama") is None:
    print("Ollama must be installed and available in your PATH.")
    print("Download it at https://ollama.com/download")
    input("Press Enter to exit.")
    sys.exit(1)

# --- Model Options ---
MODEL_OPTIONS = {
    "llama3.2:3b": {
        "label": "Llama 3.2 (Less GPU use)",
        "embed": "nomic-embed-text:v1.5"
    },
    "mistral:7b": {
        "label": "Mistral (More GPU use)",
        "embed": "nomic-embed-text:v1.5"
    }
}
DEFAULT_MODEL = "llama3.2:3b"

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

vector_store = None
text_chunks = []
current_document_name = "No document uploaded"
current_model = DEFAULT_MODEL

# --- Frontend Assets as Strings (model selector added) ---

def build_html():
    model_options_html = ''.join([
        f'<option value="{key}" {"selected" if key==DEFAULT_MODEL else ""}>{MODEL_OPTIONS[key]["label"]}</option>'
        for key in MODEL_OPTIONS
    ])
    return f"""
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NotebookLM Local</title>
    <link rel="stylesheet" href="/static/style.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
    <div class="app-container">
        <header class="app-header">
            <div class="logo">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M14 2H6C4.9 2 4 2.9 4 4V20C4 21.1 4.9 22 6 22H18C19.1 22 20 21.1 20 20V8L14 2ZM18 20H6V4H13V9H18V20Z" fill="currentColor"></path></svg>
                <span>NotebookLM</span>
            </div>
            <div class="header-controls">
                <div class="settings-container">
                    <button id="settings-btn">Settings</button>
                    <div id="settings-dropdown" class="hidden">
                        <div class="dropdown-section">Theme</div>
                        <a href="#" id="theme-light">Light</a>
                        <a href="#" id="theme-dark">Dark</a>
                    </div>
                </div>
                <div style="margin-right:24px;">
                    <select id="model-select">{model_options_html}</select>
                </div>
                <div class="user-avatar">R</div>
            </div>
        </header>

        <main class="main-content">
            <aside class="sources-panel">
                <h3>Sources</h3>
                <input type="file" id="file-upload" accept=".pdf,.txt" hidden>
                <button id="add-source-btn">+ Add</button>
                <div id="sources-list"></div>
            </aside>

            <section class="chat-panel">
                <div id="chat-welcome">
                    <div class="welcome-logo">
                        <div class="logo-stack">
                            <div class="logo-page green"></div>
                            <div class="logo-page magenta"></div>
                            <div class="logo-page blue"></div>
                        </div>
                    </div>
                    <h1 id="welcome-title">NotebookLM Local</h1>
                    <p id="welcome-subtitle">Upload a document to get started.</p>
                </div>
                <div id="chat-history"></div>
                <div class="chat-input-area">
                    <form id="chat-form">
                        <input type="text" id="chat-input" placeholder="Ask a question about your document..." autocomplete="off" disabled>
                        <button type="submit" id="send-btn" disabled>
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                        </button>
                    </form>
                    <p class="disclaimer">NotebookLM can be inaccurate. Please double-check its responses.</p>
                </div>
            </section>

            <aside class="notes-panel">
                <h3>Notes</h3>
                <div class="notes-placeholder">
                    <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                    <p>Saved notes will appear here.</p>
                    <small>This panel is a visual placeholder.</small>
                </div>
            </aside>
        </main>
    </div>
    <script>
    document.addEventListener('DOMContentLoaded', () => {{
        // Model selector in the header controls what is sent to backend
        const modelSelect = document.getElementById('model-select');
        let currentModel = modelSelect.value;
        modelSelect.addEventListener('change', function() {{
            currentModel = this.value;
        }});
        window.getSelectedModel = () => currentModel;
    }});
    </script>
    <script src="/static/script.js"></script>
</body>
</html>
"""

CSS_STYLES = """
:root {
    --bg-main: #202124;
    --bg-panel: #2d2e31;
    --bg-input: #3c4043;
    --text-primary: #e8eaed;
    --text-secondary: #9aa0a6;
    --border-color: #5f6368;
    --accent-blue: #8ab4f8;
    --accent-blue-text: #202124;
    --logo-header-color: #E8EAED;
    --dropdown-bg: #3c4043;
    --dropdown-hover-bg: #4a4e52;
}

html[data-theme="light"] {
    --bg-main: #f8f9fa;
    --bg-panel: #ffffff;
    --bg-input: #f1f3f4;
    --text-primary: #202124;
    --text-secondary: #5f6368;
    --border-color: #dadce0;
    --accent-blue: #1a73e8;
    --accent-blue-text: #ffffff;
    --logo-header-color: #5f6368;
    --dropdown-bg: #ffffff;
    --dropdown-hover-bg: #f1f3f4;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Roboto', sans-serif; background-color: var(--bg-main); color: var(--text-primary); overflow: hidden; transition: background-color 0.2s, color 0.2s; }
.app-container { display: flex; flex-direction: column; height: 100vh; }
.app-header { display: flex; justify-content: space-between; align-items: center; padding: 12px 24px; border-bottom: 1px solid var(--border-color); }
.logo { display: flex; align-items: center; gap: 8px; font-size: 1.2rem; font-weight: 500; color: var(--logo-header-color); }
.logo svg { color: var(--logo-header-color); }
.header-controls { display: flex; align-items: center; gap: 16px; }
.settings-container { position: relative; }
#settings-btn { background: none; border: 1px solid var(--border-color); color: var(--text-primary); padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 0.9rem; }
#settings-dropdown { position: absolute; top: calc(100% + 8px); right: 0; background-color: var(--dropdown-bg); border: 1px solid var(--border-color); border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); z-index: 10; width: 160px; padding: 8px 0; }
#settings-dropdown a { display: block; padding: 8px 16px; color: var(--text-primary); text-decoration: none; font-size: 0.9rem; }
#settings-dropdown a:hover { background-color: var(--dropdown-hover-bg); }
.dropdown-section { padding: 8px 16px; font-size: 0.8rem; color: var(--text-secondary); text-transform: uppercase; }
.hidden { display: none; }
.user-avatar { width: 32px; height: 32px; border-radius: 50%; background-color: #6d4db5; display: flex; justify-content: center; align-items: center; font-weight: 700; color: white; }
.main-content { display: flex; flex-grow: 1; overflow: hidden; }
.sources-panel, .notes-panel { flex: 0 0 280px; background-color: var(--bg-panel); padding: 20px; border-radius: 8px; margin: 16px; display: flex; flex-direction: column; gap: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.sources-panel h3, .notes-panel h3 { font-weight: 500; border-bottom: 1px solid var(--border-color); padding-bottom: 8px; }
#add-source-btn { width: 100%; padding: 10px; background-color: var(--bg-input); border: none; color: var(--text-primary); border-radius: 4px; cursor: pointer; font-size: 1rem; text-align: center; }
#add-source-btn:hover { background-color: #4a4e52; }
.source-item { background-color: var(--bg-input); padding: 12px; border-radius: 6px; font-size: 0.9rem; display: flex; align-items: center; gap: 8px; }
.chat-panel { flex-grow: 1; display: flex; flex-direction: column; padding: 16px 24px; max-width: 900px; margin: 0 auto; }
#chat-welcome { flex-grow: 1; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; }
.welcome-logo { margin-bottom: 24px; }
.logo-stack { position: relative; width: 60px; height: 60px; }
.logo-page { width: 40px; height: 50px; position: absolute; border-radius: 4px; border: 1px solid #5f6368; }
.logo-page.green { background-color: #34a853; transform: rotate(-15deg); top: 0; left: 0; }
.logo-page.magenta { background-color: #d62f8a; transform: rotate(0deg); top: 5px; left: 10px; z-index: 1; }
.logo-page.blue { background-color: #4285f4; transform: rotate(15deg); top: 0; left: 20px; }
#welcome-title { font-size: 2.5rem; font-weight: 500; margin-bottom: 8px; }
#welcome-subtitle { color: var(--text-secondary); font-size: 1.1rem; }
#chat-history { flex-grow: 1; overflow-y: auto; padding-right: 12px; }
.chat-message { margin-bottom: 20px; max-width: 90%; display: flex; gap: 12px; }
.chat-message .avatar { width: 32px; height: 32px; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-weight: 700; flex-shrink: 0; }
.chat-message .message-content { padding: 1px 16px; border-radius: 18px; line-height: 1.6; word-wrap: break-word; }
.message-content p:first-child { margin-top: 12px; }
.message-content p:last-child { margin-bottom: 12px; }
.message-content p { margin-bottom: 0.5em; }
.message-content h1, .message-content h2, .message-content h3 { margin-top: 1em; margin-bottom: 0.5em; }
.message-content ul, .message-content ol { padding-left: 2em; margin-bottom: 1em;}
.message-content code { background-color: rgba(0,0,0,0.1); padding: 2px 4px; border-radius: 3px; font-family: monospace; }
.user-message { margin-left: auto; flex-direction: row-reverse; }
.user-message .avatar { background-color: #6d4db5; color: white; }
.user-message .message-content { background-color: var(--accent-blue); color: var(--accent-blue-text); border-bottom-right-radius: 4px; }
.ai-message .avatar { background-color: #5f6368; color: white; }
.ai-message .message-content { background-color: var(--bg-panel); border: 1px solid var(--border-color); border-bottom-left-radius: 4px; }
.chat-input-area { padding-top: 16px; }
#chat-form { display: flex; gap: 8px; background-color: var(--bg-input); border-radius: 24px; padding: 4px; border: 1px solid var(--border-color); }
#chat-input { flex-grow: 1; background: none; border: none; outline: none; color: var(--text-primary); font-size: 1rem; padding: 10px 16px; }
#chat-input:disabled { cursor: not-allowed; }
#send-btn { background-color: var(--accent-blue); border: none; border-radius: 50%; width: 40px; height: 40px; display: flex; justify-content: center; align-items: center; cursor: pointer; color: var(--bg-main); }
#send-btn:disabled { background-color: var(--bg-input); color: var(--text-secondary); cursor: not-allowed; }
#send-btn svg { width: 20px; height: 20px; }
.disclaimer { text-align: center; font-size: 0.75rem; color: var(--text-secondary); margin-top: 12px; }
.notes-placeholder { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; height: 100%; color: var(--text-secondary); }
.notes-placeholder svg { color: #5f6368; margin-bottom: 16px; }
"""

JS_SCRIPT = """
document.addEventListener('DOMContentLoaded', () => {
    const addSourceBtn = document.getElementById('add-source-btn');
    const fileUploadInput = document.getElementById('file-upload');
    const sourcesList = document.getElementById('sources-list');
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const chatHistory = document.getElementById('chat-history');
    const chatWelcome = document.getElementById('chat-welcome');
    const modelSelect = document.getElementById('model-select');

    addSourceBtn.addEventListener('click', () => fileUploadInput.click());
    fileUploadInput.addEventListener('change', handleFileUpload);
    chatForm.addEventListener('submit', handleChatSubmit);

    async function handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        updateSourcesList(file.name, true);
        chatInput.disabled = true;
        sendBtn.disabled = true;
        chatInput.placeholder = "Processing document...";

        const formData = new FormData();
        formData.append('file', file);
        formData.append('model', modelSelect.value);

        try {
            const response = await fetch('/upload', { method: 'POST', body: formData });
            const result = await response.json();
            if (response.ok) {
                updateSourcesList(result.filename, false);
                chatInput.disabled = false;
                chatInput.placeholder = `Ask a question about ${result.filename}...`;
                chatHistory.innerHTML = '';
                chatWelcome.style.display = 'flex';
            } else { throw new Error(result.error || 'Failed to upload file.'); }
        } catch (error) {
            alert(`Error processing file: ${error.message}`);
            sourcesList.innerHTML = '';
            chatInput.placeholder = "Upload failed. Please try again.";
        } finally {
            sendBtn.disabled = chatInput.disabled;
        }
    }

    async function handleChatSubmit(event) {
        event.preventDefault();
        const query = chatInput.value.trim();
        if (!query) return;

        chatWelcome.style.display = 'none';
        appendMessage(query, 'user');
        chatInput.value = '';
        chatInput.disabled = true;
        sendBtn.disabled = true;

        const aiMessageElement = appendMessage('', 'ai');
        const contentElement = aiMessageElement.querySelector('.message-content');
        contentElement.innerHTML = '<span class="thinking-cursor"></span>';

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query,
                    model: modelSelect.value
                })
            });

            if (!response.ok) throw new Error(`Server error: ${response.statusText}`);

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            let fullResponse = "";
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const jsonStr = line.substring(6);
                        if (jsonStr) {
                            const data = JSON.parse(jsonStr);
                            if (data.token) {
                                fullResponse += data.token;
                                contentElement.innerHTML = fullResponse + '<span class="thinking-cursor"></span>';
                                chatHistory.scrollTop = chatHistory.scrollHeight;
                            }
                        }
                    }
                }
            }
            contentElement.innerHTML = fullResponse;

        } catch (error) {
            contentElement.innerHTML = `Error: ${error.message}`;
        } finally {
            chatInput.disabled = false;
            sendBtn.disabled = false;
            chatInput.focus();
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }
    }

    function updateSourcesList(filename, isLoading) {
        sourcesList.innerHTML = `
            <div class="source-item">
                <span>${filename} ${isLoading ? '(processing...)' : ''}</span>
            </div>`;
    }

    function appendMessage(text, sender) {
        const messageWrapper = document.createElement('div');
        messageWrapper.classList.add('chat-message', `${sender}-message`);
        const avatar = document.createElement('div');
        avatar.classList.add('avatar');
        avatar.textContent = sender === 'user' ? 'R' : 'AI';
        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');

        if (sender === 'user') {
            messageContent.textContent = text;
        } else {
            messageContent.innerHTML = text;
        }

        messageWrapper.appendChild(avatar);
        messageWrapper.appendChild(messageContent);
        chatHistory.appendChild(messageWrapper);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return messageWrapper;
    }
});
"""

# --- Helper for Ollama Model Check and Pull ---
def ensure_ollama_model(model_name):
    try:
        models = subprocess.check_output(["ollama", "list"], universal_newlines=True)
        if model_name not in models:
            print(f"Model {model_name} not found, pulling...")
            subprocess.check_call(["ollama", "pull", model_name])
    except Exception as e:
        print(f"Error ensuring Ollama model '{model_name}': {e}")

# --- Flask App and Backend Logic ---

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_text_from_file(file_path):
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return "".join(page.extract_text() for page in reader.pages)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    return ""

def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def create_vector_store(chunks, embed_model):
    try:
        embeddings = [ollama.embeddings(model=embed_model, prompt=chunk)['embedding'] for chunk in chunks]
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings, dtype=np.float32))
        return index
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

@app.route('/')
def index():
    return Response(build_html(), mimetype='text/html')

@app.route('/static/style.css')
def style():
    return Response(CSS_STYLES, mimetype='text/css')

@app.route('/static/script.js')
def script():
    return Response(JS_SCRIPT, mimetype='application/javascript')

@app.route('/upload', methods=['POST'])
def upload_file():
    global vector_store, text_chunks, current_document_name, current_model
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Model selection logic
    model = request.form.get('model', DEFAULT_MODEL)
    current_model = model
    embed_model = MODEL_OPTIONS.get(model, MODEL_OPTIONS[DEFAULT_MODEL])['embed']

    # Ensure model is installed (both LLM and embedding)
    ensure_ollama_model(model)
    ensure_ollama_model(embed_model)

    if file and (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            raw_text = get_text_from_file(filepath)
            text_chunks = get_text_chunks(raw_text)
            vector_store = create_vector_store(text_chunks, embed_model)
            if vector_store is None:
                 return jsonify({"error": "Failed to create vector store."}), 500
            current_document_name = file.filename
            return jsonify({"message": "File processed successfully", "filename": file.filename})
        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload a PDF or TXT file."}), 400

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('query')
    model = data.get('model', DEFAULT_MODEL)
    embed_model = MODEL_OPTIONS.get(model, MODEL_OPTIONS[DEFAULT_MODEL])['embed']
    if not query or not vector_store:
        return jsonify({"error": "Missing query or no document loaded"}), 400

    ensure_ollama_model(model)
    ensure_ollama_model(embed_model)

    def generate_response():
        try:
            query_embedding = np.array([ollama.embeddings(model=embed_model, prompt=query)['embedding']], dtype=np.float32)
            k = 3
            _, indices = vector_store.search(query_embedding, k)
            context = "\n\n---\n\n".join([text_chunks[i] for i in indices[0]])
            
            system_prompt = f"""
            You are an expert AI assistant. Your task is to answer questions based ONLY on the provided document context.
            The context is from the document '{current_document_name}'. You should format your answers using Markdown.
            Do not use any external knowledge. If the answer is not in the context, you MUST state: 'I could not find an answer in the provided document.'
            """
            full_prompt = f"Context from document:\n\n{context}\n\nUser Question: {query}"

            stream = ollama.chat(
                model=model,
                messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': full_prompt}],
                stream=True,
            )

            for chunk in stream:
                if 'content' in chunk['message']:
                    token = chunk['message']['content']
                    yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            error_message = json.dumps({'error': 'An error occurred on the server.'})
            yield f"data: {error_message}\n\n"

    return Response(generate_response(), mimetype='text/event-stream')

if __name__ == '__main__':
    import threading
    import webbrowser

    def open_browser():
        webbrowser.open("http://127.0.0.1:5000")

    threading.Timer(1.5, open_browser).start()
    print("Starting NotebookLM Local server...")
    print("Access at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
