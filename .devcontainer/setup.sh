#!/usr/bin/env bash
# .devcontainer/setup.sh
# ─────────────────────────────────────────────
# Runs automatically after the Codespace container is created.
# Sets up the Python environment and seeds the vector store.
# ─────────────────────────────────────────────
set -euo pipefail

echo "╔══════════════════════════════════════════╗"
echo "║      TN-LawMaster Dev Environment       ║"
echo "╚══════════════════════════════════════════╝"

# ── 1. Python deps ────────────────────────────
echo ""
echo "📦 Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "   ✅ Dependencies installed"

# ── 2. .env from example if not present ───────
if [ ! -f .env ]; then
  cp .env.example .env
  echo "   📋 .env created from .env.example"
  echo "   ⚠️  Edit .env and add your API key(s) before initializing the agent."
fi

# ── 3. Seed the vector store ──────────────────
echo ""
echo "📚 Seeding TCA vector store from data/ ..."
python scripts/ingest_tca.py --seed-only 2>&1 || echo "   ℹ️  Seed skipped (data/ is empty — add PDFs or run scripts/ingest_tca.py)"

# ── 4. Print quick-start instructions ─────────
cat <<'EOF'

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Quick Start
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Add your API key(s) to .env:
       nano .env

  2. Run the Streamlit UI (port 8501):
       streamlit run streamlit_app.py

  3. Or run the REST API (port 8000):
       uvicorn main_fastapi:app --reload

  4. Or use the CLI:
       python run_agent.py --backend groq --domain criminal

  5. Run unit tests:
       pytest test_production.py -v -m unit

  6. Ingest TCA documents:
       python scripts/ingest_tca.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF
