FROM python:3.11-slim-buster

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        wget \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app 
# Download the model file
RUN wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf -O /app/nomic-embed-text-v1.5.Q8_0.gguf

ENV EMBEDDER_MODEL /app/nomic-embed-text-v1.5.Q8_0.gguf
ENV EMBEDDER_CTX 8191
ENV PORT 8080

COPY . . 
RUN pip install -r requirements.txt

EXPOSE ${PORT}

CMD sh -c "fastapi run server.py --port ${PORT}"