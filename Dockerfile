FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server ./server
COPY models.py environment.py grader.py analysis.py ./
EXPOSE 8000

CMD ["uvicorn", "server.app:main", "--host", "0.0.0.0", "--port", "7860"]
