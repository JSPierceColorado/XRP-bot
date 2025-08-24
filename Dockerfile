# Simple Railway-ready image
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY worker.py /app/

# Expect COINBASE_API_KEY and COINBASE_API_SECRET at runtime (Railway Variables)
CMD ["python", "-u", "worker.py"]
