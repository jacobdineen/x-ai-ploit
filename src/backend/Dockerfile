FROM nvidia/cuda:12.2.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y \
    vim bash zsh curl git sudo python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --upgrade pip

WORKDIR /app

COPY docker_prod_requirements.txt setup.py ./
RUN pip3 install --no-cache-dir -r docker_prod_requirements.txt

COPY src/backend ./src/backend

RUN pip3 install -e .

WORKDIR /app/src/backend

CMD ["tail", "-f", "/dev/null"]
