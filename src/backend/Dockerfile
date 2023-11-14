FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Install necessary packages
RUN apt-get update && apt-get install -y \
    vim bash zsh curl git sudo python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --upgrade pip

# Set the working directory
WORKDIR /app

# Install Python dependencies
COPY docker_requirements.txt setup.py ./
RUN pip3 install --no-cache-dir -r docker_requirements.txt

# Copy the application code into the container
COPY src/backend ./src/backend

# Install the application
RUN pip3 install -e .

# Set the working directory to where your main application code is
WORKDIR /app/src/backend

CMD ["python3", "app.py"]
