# FROM python:3.9.13-slim

# RUN pip install -U pip
# RUN pip install pipenv

# WORKDIR /app

# COPY ["Pipfile", "Pipfile.lock", "./"]
# # COPY ["app", "app"]
# COPY [ "./",  "./app"]

# # Run a simple Python script to ensure Python works within the container
# RUN python -c "print('Hello from Docker container')"

# RUN pipenv install --system --deploy

# EXPOSE 9696

# # Use CMD to run the Flask development server for debugging purposes
# CMD ["python", "app/main.py"]
# ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "app.main:app" ]

FROM python:3.9.13-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

# COPY [ "Pipfile", "Pipfile.lock",  "./"]
COPY Pipfile Pipfile.lock /app/
# COPY [ "./",  "./app"]
COPY app /app/

RUN pipenv install --system --deploy

EXPOSE 9696

# ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "app.main:app" ]
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "main:app" ]


