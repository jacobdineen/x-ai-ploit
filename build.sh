#!/bin/sh

# Exit immediately if any command fails
set -e

# Check if a Dockerfile is present in the current directory
if [ ! -f Dockerfile ]; then
  echo "Error: No Dockerfile found in the current directory."
  exit 1
fi

# Set the image name (you can customize this)
IMAGE_NAME="xaiploit"

# Check if a version argument is provided
if [ -z "$1" ]; then
  echo "Error: Version argument is required."
  echo "Usage: ./build_docker_image.sh <version>"
  exit 1
else
  VERSION="$1"
fi

# Build the Docker image
sudo docker build -t "${IMAGE_NAME}:${VERSION}" .

# Print the result
echo "Docker image '${IMAGE_NAME}:${VERSION}' has been built successfully."
