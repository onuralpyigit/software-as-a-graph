# API-only Dockerfile (standalone deployment without Neo4j/frontend)
FROM python:3.11-slim

WORKDIR /app

# Copy the full repo so pyproject.toml, saag/ and api/ are available
COPY . .

# Install the saag package (non-editable) with API + Neo4j extras
RUN pip install --no-cache-dir ".[api,neo4j]"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]