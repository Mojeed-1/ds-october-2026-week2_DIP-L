FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requiremets.txt .
RUN pip install --no-cache-dir -r requiremets.txt

COPY App ./App
COPY streamlit ./streamlit
COPY models ./models

EXPOSE 8100 8300 8800

CMD ["uvicorn", "App.main:app", "--host", "0.0.0.0", "--port", "8800"]
