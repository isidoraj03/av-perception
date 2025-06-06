# docker/Dockerfile

FROM ubuntu:22.04

# Disable interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt update && apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    curl \
    git \
    build-essential

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3.11 && \
    ln -s $HOME/.local/bin/poetry /usr/local/bin/poetry

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml poetry.lock* README.md /app/
COPY yolo_pipeline /app/yolo_pipeline
COPY scripts /app/scripts
COPY tests /app/tests

# Install dependencies
RUN poetry install --no-interaction

# Optional default command
CMD ["bash"]
