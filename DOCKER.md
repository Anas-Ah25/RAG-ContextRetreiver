# Docker Deployment Guide

This guide explains how to run the RAG system backend using Docker.

## Prerequisites

- Docker installed on your system
- A valid Google Gemini API key
- Qdrant running locally or accessible via network

## Quick Start

### 2. Set up environment variables

**Important:** The `.env` file must exist in the `backend` directory before building the image, as it will be copied into the image during build.

If you don't have a `.env` file yet:

```bash
cp .env.example backend/.env
```

Edit `backend/.env` and add your Gemini API key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Build the Docker image

**Note:** The build will copy `backend/.env` into the image, so make sure it's configured before building.

```bash
docker build -t rag-backend .
```

**Performance Note:**  
- **First build**: May take 15-20 minutes due to heavy ML dependencies (transformers, PyTorch, etc.)
- **Subsequent builds**: Only seconds if you only changed code (dependencies are cached!)
- To force a clean rebuild: `docker build --no-cache -t rag-backend .`

### 4. Run the container

```bash
docker run -p 8000:8000 rag-backend
```

The container will use the `.env` file that was baked into the image during build.

To override environment variables at runtime (optional):

```bash
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_api_key \
  -e QDRANT_URL=http://your-qdrant-host:6333 \
  rag-backend
```

## Accessing the Application

Once running, the backend API will be available at:
- **API**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs

## Running Qdrant Separately

If you don't have Qdrant running, start it with Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

Then connect your backend to it by setting the `QDRANT_URL` environment variable.

## Useful Commands

```bash
# View logs from running container
docker logs -f <container-id>

# Stop the container
docker stop <container-id>

# Remove the container
docker rm <container-id>

# Rebuild after code changes
docker build -t rag-backend .

# Execute commands in the running container
docker exec -it <container-id> bash

# Run in detached mode (background)
docker run -d -p 8000:8000 rag-backend
```

## Configuration

### Environment Variables

The following environment variables can be configured in `backend/.env`:

- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `QDRANT_URL`: Qdrant database URL (optional, defaults to http://localhost:6333)

### Volumes

To persist data or mount local code for development:

```bash
# Mount data directory
docker run -p 8000:8000 \
  -v $(pwd)/backend/data:/app/data \
  rag-backend

# Mount entire backend for development (with hot-reload)
docker run -p 8000:8000 \
  -v $(pwd)/backend:/app \
  rag-backend \
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Troubleshooting

### Port already in use
If port 8000 is already in use, map to a different port:

```bash
docker run -p 8001:8000 rag-backend
```

### Cannot connect to Qdrant
- Ensure Qdrant is running and accessible
- Check the `QDRANT_URL` environment variable
- If Qdrant is on your host machine, use `host.docker.internal` instead of `localhost`:
  ```bash
  docker run -p 8000:8000 \
    -e QDRANT_URL=http://host.docker.internal:6333 \
    rag-backend
  ```

### Container keeps restarting
Check the logs for errors:
```bash
docker logs <container-id>
```

## Production Considerations

For production deployment:

1. Use specific version tags instead of `latest`
2. Set proper resource limits:
   ```bash
   docker run -p 8000:8000 \
     --memory="1g" \
     --cpus="1.0" \
     rag-backend
   ```
3. Use secrets management for API keys (Docker secrets, AWS Secrets Manager, etc.)
4. Configure proper logging drivers
5. Set up health checks and monitoring
6. Use a reverse proxy (nginx) for SSL/TLS
7. Run with a restart policy:
   ```bash
   docker run -p 8000:8000 \
     --restart unless-stopped \
     rag-backend
   ```
