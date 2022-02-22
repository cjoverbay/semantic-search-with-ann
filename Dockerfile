FROM python:3.9

WORKDIR /code

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

# This keeps the container running in idle mode.
# Can then run commands on the container using docker exec or docker-compose run
CMD ["tail", "-f", "/dev/null"]
