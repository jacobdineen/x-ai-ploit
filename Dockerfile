FROM alpine:latest
RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.8 \
    python3-pip \
    vim \
    zsh \
    curl \
    git \
    man-db \
    ucspi-tcp \
    gdb \
    netcat

WORKDIR /app

# Check the Python version
RUN python3 --version

# Copy the requirements file into the container
COPY docker_requirements.txt .

# Install the packages from the requirements file
RUN pip3 install --no-cache-dir -r docker_requirements.txt

# Copy the rest of the application code into the container
COPY . .
RUN sh -c "$(curl -L https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t robbyrussell \
    -p git \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-syntax-highlighting \
    -p https://github.com/zsh-users/zsh-completions
# Expose any ports the application might use (optional, depends on your application)
# EXPOSE <port_number>

# Set the command to run your application
CMD ["zsh"]
