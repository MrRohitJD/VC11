ARG NEW_RELIC_BASE_IMAGE=python_newrelic:latest
FROM ${NEW_RELIC_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NEW_RELIC_CONFIG_FILE=/app/newrelic.ini

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

EXPOSE 8000

CMD ["uvicorn", "app.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
