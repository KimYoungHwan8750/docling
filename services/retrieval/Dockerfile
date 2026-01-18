FROM python:3.10.19

WORKDIR /app

COPY retrieval/requirements.txt .
RUN pip install -r requirements.txt
COPY retrieval/ .

CMD ["fastapi", "wnp_opensearch.py", "--port", "50057"]