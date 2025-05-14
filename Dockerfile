FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry==1.8.2

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false && \
    poetry install --no-root --no-interaction --no-ansi

COPY . .

CMD ["poetry", "run", "python", "main.py"]

# docker build -t decisiontreeclassifier . && docker run --rm -it -p 8000:8000 decisiontreeclassifier
