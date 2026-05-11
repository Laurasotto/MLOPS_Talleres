FROM locustio/locust:2.29.0

COPY locustfile.py /home/locust/locustfile.py

WORKDIR /home/locust

EXPOSE 8089

CMD ["locust", "-f", "locustfile.py"]
