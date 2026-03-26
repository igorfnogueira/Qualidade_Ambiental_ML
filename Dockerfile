FROM python:3.11-slim

WORKDIR /app

COPY requirements.api.txt .
RUN pip install --no-cache-dir -r requirements.api.txt

COPY qa_api ./qa_api
COPY web ./web
COPY artifacts ./artifacts

ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "qa_api.main:app", "--host", "0.0.0.0", "--port", "7860"]

