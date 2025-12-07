FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2 (if you use psycopg2) and netcat for wait-for-db
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    netcat-openbsd \
 && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
# Make sure pip uses the python you want
RUN python3.11 -m pip install --no-cache-dir --upgrade pip \
    && python3.11 -m pip install --no-cache-dir -r requirements.txt

COPY preprocessing-service/src/ ./src/
COPY wait-for-db.sh /usr/local/bin/wait-for-db
RUN chmod +x /usr/local/bin/wait-for-db

EXPOSE 8000

# Default CMD waits for DB host:port args; when run locally via docker-compose the service names are used

CMD ["python3.11", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info", "--access-log"]
