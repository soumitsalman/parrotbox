FROM python:3.11-slim-buster

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        wget \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app 
COPY . . 

# Download the model file from Hugging Face and save it in the /app/models directory
RUN wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf

ENV EMBEDDER_MODEL ./nomic-embed-text-v1.5.Q8_0.gguf
ENV EMBEDDER_CTX 8191

RUN pip install -r requirements.txt

EXPOSE 8080

CMD [ "fastapi", "run" , "server.py", "--port", "8080"]