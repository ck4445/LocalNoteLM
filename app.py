

import os
import sys
import subprocess
import shutil
import faiss
import numpy as np
import ollama
import PyPDF2
import json
import webbrowser
import threading

from flask import Flask, request, jsonify, Response
from werkzeug.utils import secure_filename


# Helper for CREATE_NO_WINDOW on Windows
if sys.platform == "win32":
    CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW
else:
    CREATE_NO_WINDOW = 0



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

# Fix: Make the UPLOAD_FOLDER path absolute and based on the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

vector_store = None
text_chunks = []
source_filenames = []
current_model = DEFAULT_MODEL

def build_html():
    """Generates the main HTML content for the web interface."""
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
    <title>LocalNote</title>
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
                <span>LocalNote</span>
            </div>
            <div class="header-controls">
                <div class="model-select-wrapper">
                    <label for="model-select" id="model-label">Model</label>
                    <select id="model-select">{model_options_html}</select>
                </div>
                <div class="settings-container">
                    <button id="settings-btn">Theme</button>
                    <div id="settings-dropdown" class="hidden">
                        <a href="#" id="theme-light">Light</a>
                        <a href="#" id="theme-dark">Dark</a>
                    </div>
                </div>
                <button id="shutdown-btn" class="shutdown-btn">Shutdown App</button>
            </div>
        </header>
        <main class="main-content">
            <aside class="sources-panel">
                <h3>Sources</h3>
                <input type="file" id="file-upload" accept=".pdf,.txt" multiple hidden>
                <button id="add-source-btn">+ Add Documents</button>
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
                    <h1 id="welcome-title">LocalNote</h1>
                    <p id="welcome-subtitle">Upload documents to get started.</p>
                </div>
                <div id="chat-status" class="status-message" style="display:none;"></div>
                <div id="chat-history"></div>
                <div class="chat-input-area">
                    <form id="chat-form">
                        <input type="text" id="chat-input" placeholder="Upload a document to begin..." autocomplete="off" disabled>
                        <button type="submit" id="send-btn" disabled>
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                        </button>
                    </form>
                    <p class="disclaimer">LocalNote can be inaccurate. Please double-check its responses.</p>
                </div>
            </section>
            <aside class="notes-panel">
                <h3>Notes</h3>
                <div class="notes-placeholder">
                    <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></polyline><polyline points="10 9 9 9 8 9"></polyline></svg>
                    <p>Saved notes will appear here.</p>
                    <small>This panel is a visual placeholder.</small>
                </div>
            </aside>
        </main>
    </div>
    <script src="/static/script.js"></script>
</body>
</html>
"""

def build_css():
    """Generates the CSS styles for the web interface."""
    return """
:root {
    --bg-main: #202124; --bg-panel: #2d2e31; --bg-input: #3c4043; --text-primary: #e8eaed; --text-secondary: #9aa0a6;
    --border-color: #3c4043; --accent-blue: #8ab4f8; --accent-blue-text: #202124; --logo-header-color: #E8EAED;
    --dropdown-bg: #3c4043; --dropdown-hover-bg: #4a4e52; --model-selector-bg: #2d2e31;
    --model-selector-border: #8ab4f8; --user-message-bg: #373c5b; --ai-message-bg: #3c4043;
}
html[data-theme="light"] {
    --bg-main: #f8f9fa; --bg-panel: #ffffff; --bg-input: #f1f3f4; --text-primary: #202124; --text-secondary: #5f6368;
    --border-color: #e0e0e0; --accent-blue: #4285F4; --accent-blue-text: #ffffff; --logo-header-color: #5f6368;
    --dropdown-bg: #ffffff; --dropdown-hover-bg: #f1f3f4; --model-selector-bg: #f1f3f4;
    --user-message-bg: #e8f0fe; --ai-message-bg: #f1f3f4;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Roboto', sans-serif; background-color: var(--bg-main); color: var(--text-primary); overflow: hidden; transition: background-color 0.2s, color 0.2s; }
.app-container { display: flex; flex-direction: column; height: 100vh; }
.app-header { display: flex; justify-content: space-between; align-items: center; padding: 12px 24px; border-bottom: 1px solid var(--border-color); }
.logo { display: flex; align-items: center; gap: 8px; font-size: 1.2rem; font-weight: 500; color: var(--logo-header-color); }
.header-controls { display: flex; align-items: center; gap: 16px; }
.model-select-wrapper { display: flex; align-items: center; background-color: var(--model-selector-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 4px 12px; }
#model-label { font-size: 0.9rem; font-weight: 500; margin-right: 8px; color: var(--text-secondary); }
#model-select { background: none; border: none; color: var(--text-primary); font-size: 1rem; padding: 5px; font-weight: 500; border-radius: 6px; outline: none; appearance: none; cursor: pointer; }
#model-select:focus { box-shadow: 0 0 0 2px var(--accent-blue); }
#model-select option { background: var(--bg-input); color: var(--text-primary); }
.shutdown-btn { background: #d93025; color: #fff; padding: 8px 16px; border: none; border-radius: 8px; font-weight: 500; cursor: pointer; transition: background 0.2s; }
.shutdown-btn:hover { background: #a50e0e; }
.settings-container { position: relative; }
#settings-btn { background: none; border: 1px solid var(--border-color); color: var(--text-primary); padding: 8px 16px; border-radius: 8px; cursor: pointer; font-size: 0.9rem; }
#settings-dropdown { position: absolute; top: calc(100% + 8px); right: 0; background-color: var(--dropdown-bg); border: 1px solid var(--border-color); border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); z-index: 10; width: 100px; padding: 8px 0; }
#settings-dropdown a { display: block; padding: 8px 16px; color: var(--text-primary); text-decoration: none; font-size: 0.9rem; }
#settings-dropdown a:hover { background-color: var(--dropdown-hover-bg); }
.hidden { display: none; }
.main-content { display: flex; flex-grow: 1; overflow: hidden; }
.sources-panel, .notes-panel { flex: 0 0 280px; background-color: var(--bg-panel); padding: 20px; border-radius: 8px; margin: 16px; display: flex; flex-direction: column; gap: 16px; border: 1px solid var(--border-color); }
.sources-panel h3, .notes-panel h3 { font-weight: 500; border-bottom: 1px solid var(--border-color); padding-bottom: 8px; margin-bottom: 8px; }
#add-source-btn { width: 100%; padding: 10px; background-color: var(--accent-blue); border: none; color: var(--accent-blue-text); border-radius: 8px; cursor: pointer; font-size: 1rem; text-align: center; font-weight: 500; }
#add-source-btn:hover { opacity: 0.9; }
.source-item { background-color: var(--bg-input); padding: 12px; border-radius: 6px; font-size: 0.95rem; display: flex; align-items: center; gap: 10px; margin-bottom: 4px; color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.source-item .source-icon { font-size:1.1rem; color: var(--accent-blue); flex-shrink: 0;}
.chat-panel { flex-grow: 1; display: flex; flex-direction: column; padding: 16px 24px; max-width: 900px; margin: 0 auto; width: 100%; }
#chat-welcome { flex-grow: 1; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; }
.welcome-logo { margin-bottom: 24px; }
.logo-stack { position: relative; width: 60px; height: 60px; }
.logo-page { width: 40px; height: 50px; position: absolute; border-radius: 4px; border: 1px solid #5f6368; }
.logo-page.green { background-color: #34a853; transform: rotate(-15deg); top: 0; left: 0; }
.logo-page.magenta { background-color: #d62f8a; transform: rotate(0deg); top: 5px; left: 10px; z-index: 1; }
.logo-page.blue { background-color: #4285f4; transform: rotate(15deg); top: 0; left: 20px; }
#welcome-title { font-size: 2.5rem; font-weight: 500; margin-bottom: 8px; }
#welcome-subtitle { color: var(--text-secondary); font-size: 1.1rem; }
#chat-status.status-message { width: fit-content; max-width: 90%; margin: 14px auto 4px auto; text-align: center; background: var(--bg-input); color: var(--text-secondary); border-radius: 8px; padding: 9px 18px; font-size: 0.9rem; display: block; border: 1px solid var(--border-color); }
#chat-history { flex-grow: 1; overflow-y: auto; padding: 0 12px; }
.chat-message { margin-bottom: 24px; max-width: 100%; display: flex; gap: 12px; align-items: flex-start; }
.chat-message .avatar { width: 32px; height: 32px; border-radius: 50%; display: flex; justify-content: center; align-items: center; flex-shrink: 0; background-color: var(--bg-input); color: var(--text-secondary); }
.chat-message .avatar svg { width: 20px; height: 20px; }
.chat-message .message-content { padding: 12px 16px; border-radius: 12px; line-height: 1.6; word-wrap: break-word; font-size: 1rem; flex-grow: 1; }
.message-content p:not(:last-child) { margin-bottom: 1em; }
.message-content h1, .message-content h2, .message-content h3 { margin-top: 1em; margin-bottom: 0.5em; }
.message-content ul, .message-content ol { padding-left: 2em; margin-bottom: 1em;}
.message-content pre { background-color: rgba(0,0,0,0.18); padding: 12px; border-radius: 8px; margin: 1em 0; overflow-x: auto; }
.message-content code { background-color: rgba(0,0,0,0.18); padding: 2px 6px; border-radius: 3px; font-family: monospace; }
.user-message .message-content { background-color: var(--user-message-bg); }
.ai-message .message-content { background-color: var(--ai-message-bg); min-height: 2.5em; }
.thinking-cursor { display: inline-block; width: 10px; height: 1.2em; background-color: var(--text-primary); animation: blink 1s infinite; vertical-align: text-bottom; margin-left: 4px; }
@keyframes blink { 50% { opacity: 0; } }
.chat-input-area { padding-top: 12px; }
#chat-form { display: flex; gap: 8px; background-color: var(--bg-input); border-radius: 12px; padding: 6px; border: 1px solid var(--border-color); }
#chat-input { flex-grow: 1; background: none; border: none; outline: none; color: var(--text-primary); font-size: 1rem; padding: 10px 12px; }
#chat-input:disabled { cursor: not-allowed; }
#send-btn { background-color: var(--accent-blue); border: none; border-radius: 8px; width: 40px; height: 40px; display: flex; justify-content: center; align-items: center; cursor: pointer; color: var(--accent-blue-text); transition: background-color 0.2s;}
#send-btn:disabled { background-color: var(--bg-input); color: var(--text-secondary); cursor: not-allowed; }
#send-btn:hover:not(:disabled) { opacity: 0.9; }
#send-btn svg { width: 20px; height: 20px; }
.disclaimer { text-align: center; font-size: 0.75rem; color: var(--text-secondary); margin-top: 12px; }
.notes-placeholder { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; height: 100%; color: var(--text-secondary); }
.notes-placeholder svg { color: #5f6368; margin-bottom: 16px; }
::-webkit-scrollbar { width: 8px; background: transparent; }
::-webkit-scrollbar-thumb { background: #5f6368; border-radius: 5px; }
"""

def build_js():
    """Generates the client-side JavaScript for the web interface."""
    return """
document.addEventListener('DOMContentLoaded', () => {
    const addSourceBtn = document.getElementById('add-source-btn');
    const fileUploadInput = document.getElementById('file-upload');
    const sourcesList = document.getElementById('sources-list');
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const chatHistory = document.getElementById('chat-history');
    const chatWelcome = document.getElementById('chat-welcome');
    const chatStatus = document.getElementById('chat-status');
    const modelSelect = document.getElementById('model-select');
    const shutdownBtn = document.getElementById('shutdown-btn');
    const settingsBtn = document.getElementById('settings-btn');
    const settingsDropdown = document.getElementById('settings-dropdown');
    const themeLight = document.getElementById('theme-light');
    const themeDark = document.getElementById('theme-dark');

    const userIcon = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"></path></svg>`;
    const aiIcon = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zM9.5 16.5c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5zm5 0c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5zm2.5-4.5h-10v-2h10v2z"></path></svg>`;

    addSourceBtn.addEventListener('click', () => fileUploadInput.click());
    fileUploadInput.addEventListener('change', handleFileUpload);
    chatForm.addEventListener('submit', handleChatSubmit);

    shutdownBtn.addEventListener('click', async () => {
        if (confirm("Are you sure you want to shut down the LocalNote application?")) {
            await fetch('/shutdown', { method: 'POST' });
            document.body.innerHTML = '<h1 style="font-family: sans-serif; text-align: center; margin-top: 50px;">Application has been shut down. You can close this tab.</h1>';
        }
    });
    
    settingsBtn.addEventListener('click', () => settingsDropdown.classList.toggle('hidden'));
    document.addEventListener('click', (e) => {
        if (!settingsBtn.contains(e.target) && !settingsDropdown.contains(e.target)) {
            settingsDropdown.classList.add('hidden');
        }
    });

    themeLight.addEventListener('click', (e) => { e.preventDefault(); document.documentElement.setAttribute('data-theme', 'light'); });
    themeDark.addEventListener('click', (e) => { e.preventDefault(); document.documentElement.setAttribute('data-theme', 'dark'); });

    async function handleFileUpload(event) {
        const files = Array.from(event.target.files);
        if (!files.length) return;

        updateSourcesList(files.map(f => f.name), true);
        setChatInputState(true, "Processing documents...");
        showStatus("Embedding documents... This may take a moment.");
        chatWelcome.style.display = 'none';

        const formData = new FormData();
        files.forEach(file => formData.append('files', file));
        formData.append('model', modelSelect.value);

        try {
            const response = await fetch('/upload', { method: 'POST', body: formData });
            const result = await response.json();
            if (response.ok) {
                updateSourcesList(result.filenames, false);
                setChatInputState(false, `Ask a question about ${result.filenames.join(', ')}...`);
                chatHistory.innerHTML = ''; 
                hideStatus();
            } else { throw new Error(result.error || 'Failed to upload files.'); }
        } catch (error) {
            alert(`Error processing files: ${error.message}`);
            sourcesList.innerHTML = '';
            setChatInputState(true, "Upload failed. Please try again.");
            hideStatus();
        }
    }

    async function handleChatSubmit(event) {
        event.preventDefault();
        const query = chatInput.value.trim();
        if (!query) return;

        chatWelcome.style.display = 'none';
        appendMessage(query, 'user');
        chatInput.value = '';
        setChatInputState(true, "Thinking...");

        const aiMessageElement = appendMessage('', 'ai');
        const contentElement = aiMessageElement.querySelector('.message-content');
        
        try {
            showStatus("Searching sources...");
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, model: modelSelect.value })
            });

            if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
            
            showStatus("Generating response...");
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullResponse = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const jsonStr = line.substring(6);
                            if (jsonStr) {
                                const data = JSON.parse(jsonStr);
                                if (data.token) {
                                    fullResponse += data.token;
                                    contentElement.innerHTML = marked.parse(fullResponse + '<span class="thinking-cursor"></span>');
                                    chatHistory.scrollTop = chatHistory.scrollHeight;
                                } else if (data.error) {
                                    throw new Error(data.error);
                                }
                            }
                        } catch(e) { /* Incomplete JSON chunk, ignore */ }
                    }
                }
            }
            contentElement.innerHTML = marked.parse(fullResponse);
        } catch (error) {
            contentElement.innerHTML = `<p style="color:#d93025; padding:12px 0;">Error: ${error.message}</p>`;
        } finally {
            setChatInputState(false, "Ask another question...");
            chatInput.focus();
            chatHistory.scrollTop = chatHistory.scrollHeight;
            hideStatus();
        }
    }

    function updateSourcesList(filenames, isLoading) {
        sourcesList.innerHTML = filenames.map(filename =>
            `<div class="source-item" title="${filename}">
                <span class="source-icon">📄</span> 
                <span class="source-name">${filename} ${isLoading ? '(processing...)' : ''}</span>
            </div>`
        ).join('');
    }

    function appendMessage(text, sender) {
        const messageWrapper = document.createElement('div');
        messageWrapper.classList.add('chat-message', `${sender}-message`);
        
        const avatar = document.createElement('div');
        avatar.classList.add('avatar');
        avatar.innerHTML = sender === 'user' ? userIcon : aiIcon;

        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');
        
        if (sender === 'user') {
            const p = document.createElement('p');
            p.textContent = text;
            messageContent.appendChild(p);
        } else {
            if (text === '') {
                messageContent.innerHTML = '<span class="thinking-cursor"></span>';
            } else {
                messageContent.innerHTML = marked.parse(text);
            }
        }
        
        messageWrapper.appendChild(avatar);
        messageWrapper.appendChild(messageContent);
        chatHistory.appendChild(messageWrapper);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return messageWrapper;
    }

    function setChatInputState(disabled, placeholder) {
        chatInput.disabled = disabled;
        sendBtn.disabled = disabled;
        chatInput.placeholder = placeholder;
    }

    function showStatus(msg) { chatStatus.style.display = 'block'; chatStatus.textContent = msg; }
    function hideStatus() { chatStatus.style.display = 'none'; chatStatus.textContent = ''; }
});
"""

def ensure_ollama_model(model_name):
    """
    Checks if an Ollama model is available locally. If not, it pulls the model.
    """
    try:
        process = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if process.returncode != 0:
            print(f"Error: The 'ollama list' command failed. Is the Ollama server running?")
            print(f"Stderr: {process.stderr.strip()}")
            sys.exit(1)
        
        if model_name not in process.stdout:
            print(f"Model '{model_name}' not found locally. Pulling model, please wait...")
            pull_process = subprocess.Popen(["ollama", "pull", model_name], stdout=sys.stdout, stderr=sys.stderr)
            pull_process.wait()
            if pull_process.returncode != 0:
                print(f"Error: Failed to pull model '{model_name}'.")
                sys.exit(1)
            print(f"Model '{model_name}' pulled successfully.")
        else:
            print(f"Model '{model_name}' is available.")

    except FileNotFoundError:
        print("Error: 'ollama' command not found. Please ensure Ollama is installed and in your PATH.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while checking for Ollama models: {e}")
        sys.exit(1)


app = Flask(__name__)
@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    print("==== UNHANDLED SERVER ERROR ====")
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_text_from_file(file_path):
    if file_path.endswith('.pdf'):
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                return "".join(page.extract_text() for page in reader.pages if page.extract_text())
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""
    elif file_path.endswith('.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading TXT {file_path}: {e}")
            return ""
    return ""

def get_text_chunks(text, chunk_size=1500, chunk_overlap=250):
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def create_vector_store(all_chunks, embed_model):
    if not all_chunks:
        print("No text chunks to process for vector store.")
        return None
    try:
        print(f"Generating embeddings for {len(all_chunks)} chunks using {embed_model}...")
        embeddings = [ollama.embeddings(model=embed_model, prompt=chunk)['embedding'] for chunk in all_chunks]
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings, dtype=np.float32))
        print("Vector store created successfully.")
        return index
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

@app.route('/')
def index():
    return Response(build_html(), mimetype='text/html')

@app.route('/static/style.css')
def style():
    return Response(build_css(), mimetype='text/css')

@app.route('/static/script.js')
def script():
    return Response(build_js(), mimetype='application/javascript')

@app.route('/upload', methods=['POST'])
def upload_file():
    global vector_store, text_chunks, source_filenames, current_model
    
    if 'files' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    files = request.files.getlist('files')
    model = request.form.get('model', DEFAULT_MODEL)
    embed_model = MODEL_OPTIONS.get(model, MODEL_OPTIONS[DEFAULT_MODEL])['embed']

    all_chunks = []
    processed_filenames = []
    for file in files:
        if not file or not file.filename:
            continue
        if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
            print(f"Skipping unsupported file type: {file.filename}")
            continue

        filename = secure_filename(file.filename)
        if not filename:
            print(f"Skipping file with an invalid or insecure name: {file.filename}")
            continue
            
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            print(f"Processing file: {filename}")
            raw_text = get_text_from_file(filepath)
            if not raw_text:
                print(f"Warning: No text could be extracted from {filename}.")
                processed_filenames.append(filename)
                continue
                
            file_chunks = get_text_chunks(raw_text)
            all_chunks.extend(file_chunks)
            processed_filenames.append(filename)
        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")
            continue

    if not processed_filenames:
        return jsonify({"error": "No valid files were processed. Please upload supported file types (.pdf, .txt)."}), 400
    if not all_chunks:
        return jsonify({"error": "Could not extract any text content from the processed files. They may be empty or corrupted."}), 400

    text_chunks.clear()
    text_chunks.extend(all_chunks)
    source_filenames = processed_filenames
    current_model = model
    vector_store = create_vector_store(all_chunks, embed_model)
    if vector_store is None:
        return jsonify({"error": "Failed to create vector store from documents."}), 500
        
    return jsonify({"message": "Files processed successfully", "filenames": processed_filenames})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('query')
    model = data.get('model', current_model)
    embed_model = MODEL_OPTIONS.get(model, MODEL_OPTIONS[DEFAULT_MODEL])['embed']

    if not query:
        return jsonify({"error": "Missing query"}), 400
    if vector_store is None:
        return jsonify({"error": "No document has been loaded. Please upload a file first."}), 400
    
    def generate_response():
        print("\n--- Entering chat generator ---")
        try:
            query_embedding = np.array([ollama.embeddings(model=embed_model, prompt=query)['embedding']], dtype=np.float32)
            k = 4
            distances, indices = vector_store.search(query_embedding, k)
            context = "\n\n---\n\n".join([text_chunks[i] for i in indices[0]])
            
            print(f"Context length: {len(context)}")
            source_str = ', '.join(source_filenames) if source_filenames else "your sources"
            
            system_prompt = f"""You are a helpful and precise AI assistant called LocalNote. 
            Your task is to answer questions based *only* on the provided document context.
            The context is from the following document(s): {source_str}.
            - Format your answers clearly using Markdown.
            - Be concise and directly answer the user's question.
            - If the answer is not found within the provided context, you MUST state: 'I could not find an answer in the provided document(s).'
            - Do not use any external knowledge or make up information.
            """
            full_prompt = f"CONTEXT FROM DOCUMENTS:\n\n{context}\n\nUSER QUESTION: {query}"

            stream = ollama.chat(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': full_prompt}
                ],
                stream=True,
            )

            print("Ollama stream started. Waiting for chunks...")
            for i, chunk in enumerate(stream):
                if i == 0:
                    print("First chunk received from Ollama.")
                if 'content' in chunk['message']:
                    token = chunk['message']['content']
                    # FIX: Use single newline for proper SSE formatting.
                    yield f"data: {json.dumps({'token': token})}\n\n"

        except Exception as e:
            print(f"Error during chat generation: {e}")
            error_message = json.dumps({'error': 'An error occurred on the server while generating the response.'})
            # FIX: Use single newline for proper SSE formatting.
            yield f"data: {error_message}\n\n"

    return Response(generate_response(), mimetype='text/event-stream')

@app.route('/shutdown', methods=['POST'])
def shutdown():
    print("Shutdown request received. Terminating server.")
    os._exit(0) 

if __name__ == '__main__':
    if getattr(sys, 'frozen', False):
        os.chdir(sys._MEIPASS)
        
    print("--- LocalNote Startup Check ---")
    for model_key in MODEL_OPTIONS:
        ensure_ollama_model(model_key)
        ensure_ollama_model(MODEL_OPTIONS[model_key]['embed'])
    print("--- All models are ready. ---")
    
    url = "http://127.0.0.1:5000"
    threading.Timer(1.25, lambda: webbrowser.open(url)).start()
    
    print("\nStarting LocalNote server...")
    print(f"Access at {url}")
    app.run(host='127.0.0.1', port=5000, threaded=True)