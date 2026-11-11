FROM python:3.12-slim

RUN pip install --no-cache-dir "newrelic>=10,<11"

ENTRYPOINT ["newrelic-admin", "run-program"]
