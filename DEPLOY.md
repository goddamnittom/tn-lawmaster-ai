# TN-LawMaster — Deployment Guide

Comprehensive deployment instructions for all supported platforms.

---

## 🚀 Option 1: Fly.io (Recommended — Free Tier)

[Fly.io](https://fly.io) gives you a free permanent URL, HTTPS, and a persistent volume for ChromaDB.

### Prerequisites
```bash
# Install flyctl
# Windows (PowerShell):
iwr https://fly.io/install.ps1 -useb | iex
# macOS/Linux:
curl -L https://fly.io/install.sh | sh

fly auth login
```

### Deploy
```bash
cd C:\Users\blake\tn-lawmaster-ai

# First-time deployment (creates the app and volume)
fly launch --name tn-lawmaster-ai --region iad --no-deploy

# Set your secrets (never commit these)
fly secrets set ACTIVE_BACKEND=groq
fly secrets set GROQ_API_KEY=your_key_here
fly secrets set API_KEY=your_api_secret_here     # optional: protect your endpoints

# Create the persistent volume for ChromaDB (1GB free)
fly volumes create tn_lawmaster_data --size 1 --region iad

# Deploy
fly deploy
```

### Verify
```bash
fly status
curl https://tn-lawmaster-ai.fly.dev/health
```

### Update after code changes
```bash
fly deploy
```

---

## 🐳 Option 2: Docker Compose (Self-hosted)

### Single machine (VPS, home server)
```bash
# Clone & configure
git clone https://github.com/goddamnittom/tn-lawmaster-ai.git
cd tn-lawmaster-ai
cp .env.example .env
nano .env   # fill in your API keys

# Build and run
docker compose up -d --build

# View logs
docker compose logs -f

# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### With Nginx reverse proxy (production)
```bash
# Start with the production compose file
docker compose -f tn-lawmaster-production/docker-compose.yml up -d

# The Nginx config handles SSL termination
# Update nginx/nginx.conf with your domain
```

---

## 🚂 Option 3: Railway.app (Zero-config)

1. Fork the repo on GitHub
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
3. Select your fork
4. Add environment variables in the Railway dashboard:
   - `ACTIVE_BACKEND` = `groq`
   - `GROQ_API_KEY` = your key
   - `API_KEY` = your secret (optional)
   - `PORT` = `8000`
5. Railway auto-detects the Dockerfile and deploys

**Note**: Railway's free tier has usage limits. Use a paid plan for production.

---

## ☁️ Option 4: Google Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/YOUR_PROJECT/tn-lawmaster-ai

# Deploy
gcloud run deploy tn-lawmaster-ai \
  --image gcr.io/YOUR_PROJECT/tn-lawmaster-ai \
  --platform managed \
  --region us-east1 \
  --port 8000 \
  --memory 1Gi \
  --set-env-vars ACTIVE_BACKEND=groq \
  --set-secrets GROQ_API_KEY=groq-api-key:latest

# URL will be printed after deployment
```

---

## 🌐 Streamlit Cloud (UI only)

Deploy the Streamlit UI to [share.streamlit.io](https://share.streamlit.io) (free):

1. Fork the repo
2. Go to share.streamlit.io → **New app**
3. Select your fork, branch `main`, file `streamlit_app.py`
4. Add secrets in the Streamlit Cloud dashboard (Settings → Secrets):
   ```toml
   ACTIVE_BACKEND = "groq"
   GROQ_API_KEY = "your_key_here"
   ```
5. Deploy

**Note**: Streamlit Cloud works best with cloud backends (Groq, OpenAI, OpenRouter).
Ollama requires a local server and won't work on Streamlit Cloud.

---

## 🔐 Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `ACTIVE_BACKEND` | Yes | `ollama` \| `groq` \| `openai` \| `openrouter` |
| `GROQ_API_KEY` | If using Groq | [console.groq.com](https://console.groq.com) |
| `OPENAI_API_KEY` | If using OpenAI | [platform.openai.com](https://platform.openai.com) |
| `OPENROUTER_API_KEY` | If using OpenRouter | [openrouter.ai](https://openrouter.ai) |
| `OLLAMA_BASE_URL` | If using Ollama | Default: `http://localhost:11434` |
| `API_KEY` | Optional | Protect your endpoints with a secret key |
| `CORS_ORIGINS` | Optional | Comma-separated allowed origins (default: `*`) |
| `CHROMA_DB_PATH` | Optional | ChromaDB persist path (default: `./chroma_db`) |
| `CACHE_TTL_SECONDS` | Optional | Response cache TTL (default: `3600`) |
| `MEMORY_DB_PATH` | Optional | SQLite memory path (default: `./memory.db`) |
| `LOG_LEVEL` | Optional | `DEBUG`\|`INFO`\|`WARNING` (default: `INFO`) |

---

## 📊 Setting Secrets on Each Platform

### Fly.io
```bash
fly secrets set GROQ_API_KEY=sk-xxx API_KEY=my-secret
fly secrets list   # verify
```

### Railway
Dashboard → Variables tab → Add variable

### Google Cloud Run
```bash
# Store in Secret Manager first
echo -n "sk-xxx" | gcloud secrets create groq-api-key --data-file=-
# Then reference in deploy command with --set-secrets
```

### Docker
```bash
# Use .env file (never commit it)
docker run --env-file .env tn-lawmaster-ai
# Or individual -e flags:
docker run -e GROQ_API_KEY=sk-xxx -e ACTIVE_BACKEND=groq tn-lawmaster-ai
```

---

## ✅ Post-Deployment Checklist

1. `curl https://your-url/health` returns `{"status": "healthy"}`
2. `curl https://your-url/docs` shows Swagger UI
3. Seed the vector store: `curl -X POST https://your-url/ingest/text -H "Content-Type: application/json" -d '{"text": "TCA § 39-14-103...", "source": "TCA § 39-14-103"}'`
4. Run a test query: `curl -X POST https://your-url/analyze -H "Content-Type: application/json" -d '{"query": "DUI penalty in Tennessee", "domain": "traffic"}'`
5. If API_KEY is set, add `-H "X-API-Key: your-key"` to all requests

---

## 🩺 Monitoring

### Fly.io
```bash
fly status          # machine status
fly logs            # real-time logs
fly metrics         # CPU/memory
```

### Health endpoint
```bash
watch -n 30 'curl -s https://your-url/health | python3 -m json.tool'
```
