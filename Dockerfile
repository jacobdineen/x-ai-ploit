# Use the official Ubuntu Docker image as the base image
FROM ubuntu:latest

# Set the working directory in the container
WORKDIR /app

# Update the package list and install Python
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y python3 python3-pip

# Check the Python version
RUN python3 --version

# Copy the requirements file into the container
COPY docker_requirements.txt .

# Install the packages from the requirements file
RUN pip3 install --no-cache-dir -r docker_requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose any ports the application might use (optional, depends on your application)
# EXPOSE <port_number>

# Set the command to run your application
# CMD ["python3", "your_application.py"]
