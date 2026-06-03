FROM python:3.11-slim

RUN pip install --no-cache-dir "importlib-metadata>=3.7.0,<8,!=4.7.0" && \
    pip install --no-cache-dir \
    mlflow==2.13.0 \
    psycopg2-binary \
    boto3

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5000

ENTRYPOINT ["/entrypoint.sh"]
