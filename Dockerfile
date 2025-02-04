FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including OpenGL
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir watchdog

COPY . .

ENV PYTHONPATH=/app/src
ENV PORT=8501

EXPOSE 8501

CMD streamlit run src/pages/dashboard.py --server.port=$PORT --server.address=0.0.0.0
