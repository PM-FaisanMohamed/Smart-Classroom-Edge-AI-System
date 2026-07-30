FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for OpenCV and YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxcb1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (uses cached layer unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and dashboards
COPY backend/ ./backend/
COPY dashboard/ ./dashboard/
COPY ac-simulation/ ./ac-simulation/
COPY models/ ./models/

# Expose server port
EXPOSE 5000

# Run FastAPI backend with Uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "5000"]
