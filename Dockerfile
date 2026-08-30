FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.app:app --host --port ${PORT:-8000} --reload"]