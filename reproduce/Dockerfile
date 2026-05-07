FROM python:3.11-slim

LABEL org.opencontainers.image.title="Q-HGL Middleware 2026"
LABEL org.opencontainers.image.description=\
    "QoS-Aware Heterogeneous Graph Learning — reproducibility container"
LABEL org.opencontainers.image.authors="Q-HGL Authors"

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy requirements first (Docker cache layer)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install PyTorch Geometric (CPU-only build for reproducibility)
RUN pip install --no-cache-dir \
        torch==2.2.1+cpu \
        torch-scatter \
        torch-sparse \
        torch-geometric \
        -f https://data.pyg.org/whl/torch-2.2.1+cpu.html || \
    pip install --no-cache-dir torch-geometric

# ── Project source ─────────────────────────────────────────────────────────────
COPY . .

# Ensure project root is on PYTHONPATH
ENV PYTHONPATH=/workspace

# ── Verify installation ────────────────────────────────────────────────────────
RUN python -c "from saag.prediction.data_preparation import EDGE_FEATURE_DIM; \
               assert EDGE_FEATURE_DIM == 16, f'Expected 16, got {EDGE_FEATURE_DIM}'; \
               print(f'✓ EDGE_FEATURE_DIM={EDGE_FEATURE_DIM}')" && \
    python -c "from saag.prediction.models import NodeCriticalityGNN; print('✓ Models OK')"

# ── Default entrypoint: smoke-test → full pipeline ─────────────────────────────
CMD ["make", "-C", "/workspace/reproduce", "all", \
     "PYTHON=python", "EPOCHS=300", "SEEDS=42,123,456,789,2024"]
