# Lawracle – AI Legal Assistant (RAG-LLM)

Lawracle is an AI-powered legal assistant specializing in Indian law. It utilizes **Mistral-7B / TinyLlama** combined with **LlamaIndex** and **ChromaDB** for Retrieval-Augmented Generation (RAG).

---

## How to Start Locally

You can run this project locally using either **Docker Compose** (recommended) or **Natively**.

### Option 1: Using Docker Compose (Recommended)

This method spins up both the Python Flask backend and the Nginx frontend in containerized environments. It automatically handles URL proxying exactly as configured for production.

1. **Start the containers**:
   ```bash
   docker compose up --build
   ```
2. **Access the application**:
   - Open your browser and navigate to: **[http://localhost:8081](http://localhost:8081)**
3. **Adding documents**:
   - Place your PDF, TXT, or MD documents into the `./data` directory. The backend will automatically parse and vectorize them on startup.

*Note: The first startup will download the `TinyLlama` model (~2.2GB), which will be cached in a Docker volume for subsequent starts.*

---

### Option 2: Running Natively

If you prefer not to use Docker, you can run the Flask backend and serve the frontend directly.

#### 1. Start the Flask Backend
1. Make sure you have **Python 3.10+** installed.
2. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the backend:
   ```bash
   python app.py
   ```
   *The backend will start running on [http://127.0.0.1:5000](http://127.0.0.1:5000).*

#### 2. Open the Frontend
1. Double-click the `frontend/index.html` file to open it directly in your browser (`file:///...`), or serve it using a lightweight server (e.g. `npx serve` or `python -m http.server`).
2. The frontend is configured to automatically route API requests to `http://127.0.0.1:5000` when running locally.