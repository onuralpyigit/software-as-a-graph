# Use a lightweight Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    MPLCONFIGDIR=/tmp/matplotlib

# ck tool version (Java OO metrics: size, complexity, cohesion)
ARG CK_VERSION=0.7.0

# Install system dependencies
# - git: Required for cloning repositories
# - make: Required for running the pipeline (Makefile)
# - curl: Required for downloading ck JAR
# - openjdk-21-jre-headless: Required by ck (Java tool)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    make \
    curl \
    openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Install ck (Code metrics for Java) - single JAR, no server needed
# Downloaded from Maven Central (no GitHub Release assets available)
RUN mkdir -p /opt/ck \
    && curl -fsSL -o /opt/ck/ck.jar \
       "https://repo1.maven.org/maven2/com/github/mauricioaniche/ck/${CK_VERSION}/ck-${CK_VERSION}-jar-with-dependencies.jar"

ENV CK_JAR_PATH="/opt/ck/ck.jar"

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create output directory
RUN mkdir -p output

# Default entrypoint
CMD ["/bin/bash"]
