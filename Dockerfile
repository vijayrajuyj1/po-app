FROM python:3.12-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required by OpenCV and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast dependency manager)
RUN pip install --no-cache-dir uv

# Copy dependency file first to enable layer caching
COPY pyproject.toml ./
# If you have uv.lock include it to speed up build
# COPY uv.lock ./

# Install torch (separately because it's heavy and external index)
RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu

# Now install everything from pyproject
RUN uv pip install --system "."

# Copy the rest of the app
COPY . .

EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
