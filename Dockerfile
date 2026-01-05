FROM python:3.11-slim

WORKDIR /app

#TODO: Installing dependencies
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p logs

#! Exposing the streamlit port
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
