# LLM Fine-Tuning UI - Docker Setup

This document provides comprehensive instructions for setting up and running the LLM Fine-Tuning UI using Docker.

## 🐳 Prerequisites

### Required Software
- **Docker** (v20.10+)
- **Docker Compose** (v2.0+)
- **NVIDIA Docker Runtime** (for GPU support)
- **NVIDIA Drivers** (latest recommended)

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (tested on dual RTX 3060)
- **RAM**: Minimum 16GB, recommended 32GB+
- **Storage**: At least 50GB free space for models and data

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd llm-trainer-ui
```

### 2. Run Setup Script
```bash
./docker-setup.sh dev
```

This will:
- Check prerequisites
- Create necessary directories
- Build Docker images
- Start development services

### 3. Access the Application
- **Frontend**: http://localhost:55155
- **Backend API**: http://localhost:8001
- **Health Check**: http://localhost:8001/health

## 📋 Available Commands

### Using the Setup Script
```bash
# Development environment
./docker-setup.sh dev

# Production environment
./docker-setup.sh prod

# Stop all services
./docker-setup.sh stop

# View logs
./docker-setup.sh logs

# Check status
./docker-setup.sh status

# Cleanup resources
./docker-setup.sh cleanup
```

### Using Docker Compose Directly

#### Development
```bash
# Start development environment
docker-compose -f docker-compose.yml -f docker-compose.dev.yml --profile development up -d

# Stop development environment
docker-compose -f docker-compose.yml -f docker-compose.dev.yml --profile development down
```

#### Production
```bash
# Start production environment
docker-compose --profile production up -d

# Stop production environment
docker-compose --profile production down
```

## 🏗️ Architecture

### Services

#### Backend (`llm-trainer-backend`)
- **Base Image**: `nvidia/cuda:12.1-devel-ubuntu22.04`
- **Port**: 8001
- **GPU Access**: Full access to all available GPUs
- **Volumes**: 
  - `./backend/uploads:/app/uploads`
  - `./backend/logs:/app/logs`
  - `./backend/checkpoints:/app/checkpoints`
  - `./backend/configs:/app/configs`

#### Frontend Development (`llm-trainer-frontend-dev`)
- **Base Image**: `node:18-alpine`
- **Port**: 55155
- **Hot Reload**: Enabled with volume mounting
- **API Proxy**: Configured to proxy `/api/*` to backend

#### Frontend Production (`llm-trainer-frontend-prod`)
- **Base Image**: `nginx:alpine`
- **Port**: 80
- **Features**: Optimized build, gzip compression, API proxying

### Networks
- **llm-trainer-network**: Bridge network for service communication

### Volumes
- **Persistent Data**: `uploads/`, `logs/`, `checkpoints/`, `configs/`
- **Node Modules**: Anonymous volume for frontend dependencies

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
# Backend Configuration
DEBUG=false
LOG_LEVEL=info
CUDA_VISIBLE_DEVICES=0,1

# Frontend Configuration
VITE_API_URL=http://localhost:8001

# Model Storage
HUGGINGFACE_HUB_CACHE=/app/cache/huggingface
TRANSFORMERS_CACHE=/app/cache/transformers
```

### GPU Configuration

The setup automatically detects and uses all available NVIDIA GPUs. To limit GPU usage:

```yaml
# In docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=0  # Use only GPU 0
```

## 🔧 Development Workflow

### 1. Start Development Environment
```bash
./docker-setup.sh dev
```

### 2. Code Changes
- **Backend**: Changes auto-reload with uvicorn `--reload`
- **Frontend**: Hot module replacement via Vite

### 3. View Logs
```bash
# All services
./docker-setup.sh logs

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend-dev
```

### 4. Debug
```bash
# Access backend container
docker exec -it llm-trainer-backend bash

# Access frontend container
docker exec -it llm-trainer-frontend-dev sh
```

## 🏭 Production Deployment

### 1. Build and Start
```bash
./docker-setup.sh prod
```

### 2. Production Features
- **Optimized Frontend**: Minified and compressed
- **Nginx Proxy**: Handles static files and API routing
- **Health Checks**: Automatic container health monitoring
- **Resource Limits**: Configured for production workloads

### 3. SSL/HTTPS Setup
Add SSL configuration to `frontend/nginx.conf`:

```nginx
server {
    listen 443 ssl;
    ssl_certificate /etc/ssl/certs/your-cert.pem;
    ssl_certificate_key /etc/ssl/private/your-key.pem;
    # ... rest of configuration
}
```

## 📊 Monitoring

### Service Health
```bash
# Check all services
./docker-setup.sh status

# Backend health
curl http://localhost:8001/health

# Container stats
docker stats
```

### GPU Monitoring
```bash
# Inside backend container
docker exec llm-trainer-backend nvidia-smi

# Continuous monitoring
watch -n 1 'docker exec llm-trainer-backend nvidia-smi'
```

### Logs
```bash
# Real-time logs
./docker-setup.sh logs

# Specific timeframe
docker-compose logs --since="1h" --until="30m"
```

## 🔒 Security Considerations

### Production Security
1. **Change Default Secrets**: Update `SECRET_KEY` in `.env`
2. **Network Security**: Use proper firewall rules
3. **Volume Permissions**: Ensure proper file permissions
4. **Regular Updates**: Keep base images updated

### GPU Security
- Containers have full GPU access
- Consider resource limits for multi-tenant deployments

## 🐛 Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi

# Install nvidia-docker2 if missing
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

#### Out of Memory
```bash
# Check GPU memory
docker exec llm-trainer-backend nvidia-smi

# Reduce model size or use quantization
# Edit backend configuration for smaller models
```

#### Port Conflicts
```bash
# Check port usage
sudo lsof -i :8001
sudo lsof -i :55155

# Modify ports in docker-compose.yml if needed
```

#### Build Failures
```bash
# Clean build cache
docker system prune -a

# Rebuild from scratch
docker-compose build --no-cache
```

### Performance Tuning

#### GPU Optimization
- Use appropriate precision (fp16/bf16)
- Enable gradient checkpointing for large models
- Use model parallelism for multi-GPU setups

#### Storage Optimization
- Use SSD for model storage
- Consider NFS for shared model storage
- Implement model caching strategies

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [FastAPI Docker Guide](https://fastapi.tiangolo.com/deployment/docker/)
- [Vite Docker Guide](https://vitejs.dev/guide/build.html#docker)

## 🆘 Support

For issues and questions:
1. Check the logs: `./docker-setup.sh logs`
2. Verify GPU access: `docker exec llm-trainer-backend nvidia-smi`
3. Check service status: `./docker-setup.sh status`
4. Review this documentation
5. Open an issue in the repository 