# CorruptionReportingResearch
Corruption Reporting Academic Research Experiment

# Start production
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Scale backend
docker-compose -f docker/docker-compose.yml up -d --scale backend=3

# Stop
docker-compose -f docker/docker-compose.yml down

# Backup
docker-compose -f docker/docker-compose.yml exec backend python /app/scripts/backup.py

# Start development
docker-compose -f docker/docker-compose.dev.yml up

# View specific logs
docker-compose -f docker/docker-compose.dev.yml logs -f backend

# Run tests
docker-compose -f docker/docker-compose.dev.yml --profile testing up test

# Interactive shell
docker-compose -f docker/docker-compose.dev.yml exec backend bash
docker-compose -f docker/docker-compose.dev.yml exec frontend sh

# Rebuild after dependency changes
docker-compose -f docker/docker-compose.dev.yml up --build

# Stop
docker-compose -f docker/docker-compose.dev.yml down

# Inspect volumes
docker volume ls
docker volume inspect docker_backend-data

# Backup volume
docker run --rm -v docker_backend-data:/data -v $(pwd):/backup alpine tar czf /backup/data-backup.tar.gz -C /data .

# Restore volume
docker run --rm -v docker_backend-data:/data -v $(pwd):/backup alpine tar xzf /backup/data-backup.tar.gz -C /data


# 1. Build images
docker-compose -f docker/docker-compose.yml build

# 2. Initialize storage
docker-compose -f docker/docker-compose.yml run --rm backend python /app/scripts/initialize_storage.py

# 3. Seed validators
docker-compose -f docker/docker-compose.yml run --rm backend python /app/scripts/seed_validators.py

# 4. Start services
docker-compose -f docker/docker-compose.yml up -d

# 5. Check health
docker-compose -f docker/docker-compose.yml ps
curl http://localhost:8000/health
curl http://localhost:3000/health

# 1. Build and tag
docker-compose -f docker/docker-compose.yml build
docker tag corruption-backend:latest myregistry.azurecr.io/corruption-backend:latest
docker tag corruption-frontend:latest myregistry.azurecr.io/corruption-frontend:latest

# 2. Push to ACR
az acr login --name myregistry
docker push myregistry.azurecr.io/corruption-backend:latest
docker push myregistry.azurecr.io/corruption-frontend:latest

# 3. Deploy with docker-compose
docker context create aci myaci --resource-group myResourceGroup
docker context use myaci
docker compose -f docker/docker-compose.yml up

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Service logs
docker-compose -f docker/docker-compose.yml logs -f backend
docker-compose -f docker/docker-compose.yml logs -f frontend

# Resource usage
docker-compose -f docker/docker-compose.yml stats

# Process list
docker-compose -f docker/docker-compose.yml top

# Execute commands
docker-compose -f docker/docker-compose.yml exec backend python /app/scripts/health_check.py
docker-compose -f docker/docker-compose.yml exec backend python /app/scripts/backup.py


.env example:
# Application
CONFIG_ENV=production
LOG_LEVEL=info

# Backend
MAX_WORKERS=2
TIMEOUT=120

# Frontend
NODE_ENV=production
BACKEND_URL=http://backend:8000

# Ports
BACKEND_PORT=8000
FRONTEND_PORT=3000

Network Architecture:
┌─────────────────────────────────────┐
│  Host Machine                       │
│  ┌───────────────┐  ┌─────────────┐│
│  │ localhost:3000│  │localhost:8000││
│  └───────┬───────┘  └──────┬──────┘│
│          │                 │        │
│  ┌───────▼─────────────────▼──────┐│
│  │   corruption-network          ││
│  │  ┌──────────┐  ┌────────────┐ ││
│  │  │ frontend │──│  backend   │ ││
│  │  │  :3000   │  │   :8000    │ ││
│  │  └──────────┘  └────────────┘ ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
