FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY customerlens ./customerlens
COPY dashboard.py ./dashboard.py
COPY data ./data

# Build the retrieval artifacts at image build time so the API starts ready.
RUN python -m customerlens.cli ingest \
    --tickets data/tickets.csv --customers data/customers.csv --out data/artifacts

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=4).read() else 1)"

CMD ["python", "-m", "customerlens.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
