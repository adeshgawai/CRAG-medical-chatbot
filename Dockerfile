# Use python 3.11 slim as the base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for extremely fast dependency installation
RUN pip install --no-cache-dir uv

# Copy dependency definition files
COPY pyproject.toml uv.lock ./

# Install project dependencies globally in the container
RUN uv pip install --system -r pyproject.toml

# Copy the rest of the application files
COPY . .

# Expose Hugging Face Space port
EXPOSE 7860

# Run the Flask server
CMD ["python", "app.py"]
