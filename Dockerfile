FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY models ./models

EXPOSE 5000

CMD ["python", "api/app.py"]
