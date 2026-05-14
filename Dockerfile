# Python Base Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libportaudio2 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY Automatic_detector.py .
COPY yolov8n.pt .
COPY README.md .
COPY "face id/" "face id/"

# Create directories for data
RUN mkdir -p /app/data /app/temp

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:0

# Default command
CMD ["python", "Automatic_detector.py"]
