# 🚀 AI Data Cleaner - Deployment Guide

This guide explains how to deploy the AI Data Cleaner API and Interactive Dashboard to production environments using **Docker**, **Render**, **Railway**, **Hugging Face Spaces**, or **Cloud Virtual Machines (AWS / GCP / DigitalOcean)**.

---

## 📋 Prerequisites & Environment Variables

| Variable | Required | Description |
|---|---|---|
| `PORT` | Optional (default: `8000`) | Server listening port |
| `GEMINI_API_KEY` | Optional | Google Gemini API key (defaults to free-tier `gemini-2.5-flash`) |
| `OPENAI_API_KEY` | Optional | OpenAI API key (for GPT-4o-mini or Groq/vLLM endpoints) |
| `ANTHROPIC_API_KEY` | Optional | Anthropic API key (for Claude 3.5 Haiku / Sonnet) |
| `OLLAMA_BASE_URL` | Optional | Local/air-gapped Ollama URL (default: `http://localhost:11434`) |

> [!NOTE]
> Even if no LLM API key is provided, the dashboard still functions completely using the built-in deterministic **Mock Provider** and rule-based heuristic guardrails.

---

## 🐳 Option 1: Docker & Docker Compose (Recommended)

### Using Docker Compose
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/Ok-Sanjeev/ai-data-cleaner.git
   cd ai-data-cleaner
   ```
2. (Optional) Set your API keys in `.env` or environment:
   ```bash
   export GEMINI_API_KEY="your-gemini-key"
   ```
3. Build and run the container:
   ```bash
   docker compose up -d --build
   ```
4. Access the dashboard:
   👉 **`http://localhost:8000`**

### Using Standalone Docker
```bash
# Build the Docker image
docker build -t ai-data-cleaner .

# Run the container
docker run -d -p 8000:8000 \
  -e GEMINI_API_KEY="your-key" \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/runs:/app/runs \
  --name ai-cleaner ai-data-cleaner
```

---

## 🟣 Option 2: 1-Click Deploy on Render.com

1. Fork or push this repository to your GitHub account.
2. Sign in to [Render.com](https://render.com) and click **New + > Blueprint**.
3. Select your repository. Render will automatically detect `render.yaml`.
4. Add your `GEMINI_API_KEY` in the Environment Variables section.
5. Click **Apply**. Your app will build and deploy on a free or standard Render web service with automatic HTTPS!

---

## 🚂 Option 3: 1-Click Deploy on Railway.app

1. Sign in to [Railway.app](https://railway.app).
2. Click **New Project > Deploy from GitHub repo**.
3. Select `ai-data-cleaner`. Railway detects `railway.json` and `Dockerfile`.
4. Under **Variables**, add `GEMINI_API_KEY` (and optional `OPENAI_API_KEY`).
5. Click **Deploy**. Railway provisions a live public domain (e.g. `https://ai-data-cleaner-production.up.railway.app`).

---

## 🤗 Option 4: Deploy on Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Choose **Docker** as the Space SDK and blank template.
3. Push the contents of this repository to the Space's Git repository.
4. In Space **Settings > Variables and secrets**, add your `GEMINI_API_KEY` secret.
5. Hugging Face will automatically build the `Dockerfile` and serve the dashboard on port `8000`.

---

## 💻 Option 5: Deploy on Linux VM (Ubuntu / Debian / AWS EC2)

1. Connect to your server and clone:
   ```bash
   git clone https://github.com/Ok-Sanjeev/ai-data-cleaner.git /opt/ai-data-cleaner
   cd /opt/ai-data-cleaner
   ```
2. Setup Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Create a systemd service (`/etc/systemd/system/aicleaner.service`):
   ```ini
   [Unit]
   Description=AI Data Cleaner Service
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/opt/ai-data-cleaner
   ExecStart=/opt/ai-data-cleaner/venv/bin/python server.py --host 0.0.0.0 --port 8000
   Restart=always
   Environment=PORT=8000
   Environment=GEMINI_API_KEY=your-api-key

   [Install]
   WantedBy=multi-user.target
   ```
4. Start and enable service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable aicleaner
   sudo systemctl start aicleaner
   ```

---

## 🔒 Security & Resource Controls

DataRefine includes built-in security hardening in [`server.py`](server.py):

- **Strict File Root Jail (`ALLOWED_DATA_DIRS`)**: Client-supplied dataset file paths are validated by `resolve_safe_dataset_path()` and strictly confined to approved `data/` subdirectories (`data`, `data/uploads`, `data/benchmark`, `data/robustness_tests`), blocking path traversal and arbitrary file reads.
- **Upload & Dataset Limits**: Maximum upload size is enforced at 50 MB (`MAX_UPLOAD_SIZE_BYTES`), and profiled datasets are bounded to `500,000` rows (`MAX_DATASET_ROWS`) and `500` columns (`MAX_DATASET_COLUMNS`) to guard against memory exhaustion.
- **CORS Policy (`ALLOWED_ORIGINS`)**: Defaults to wildcard `*` with `allow_credentials=False`. For production domains requiring credentials, explicitly configure `ALLOWED_ORIGINS` as a comma-separated list of trusted origins.
- **Single-Node In-Memory Sessions**: Active workspace state is held in memory (`SESSIONS`, max `50` sessions / `4 hours` TTL). Multi-worker or multi-tenant cloud deployments require sticky sessions or an external session store (e.g., Redis) plus authentication middleware.

---

## 🔍 Healthcheck & Verification

Verify the deployment is healthy:
```bash
curl -f http://localhost:8000/api/health
```
Response:
```json
{
  "status": "healthy",
  "service": "AI Data Cleaner",
  "version": "2.0.0",
  "supported_providers": ["gemini", "openai", "anthropic", "ollama", "mock"]
}
```
Interactive API docs are available at **`/docs`** (Swagger UI).
