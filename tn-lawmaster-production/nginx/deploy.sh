#!/bin/bash
set -e
echo "🚀 TN-LawMaster Production Deployment"
if [ ! -f .env ]; then cp .env.example .env; echo "Edit .env first"; exit 1; fi
export $(grep -v '^#' .env | xargs)
mkdir -p data/tn_law_db data/uploads logs/nginx
docker-compose build --no-cache
docker-compose up -d
echo "✅ Deployment complete. Check: docker-compose ps"