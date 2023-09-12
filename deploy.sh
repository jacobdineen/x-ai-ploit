#!/bin/sh
# ./deploy_docker_container.sh v1.0.0
# Set the image name (you can customize this)
IMAGE_NAME="homecams"
# Set the container name
CONTAINER_NAME="test"

# Check if a version argument is provided
if [ -z "$1" ]; then
  echo "Error: Version argument is required."
  echo "Usage: ./deploy_docker_container.sh <version>"
  exit 1
else
  VERSION="$1"
fi



# if [ "$(sudo docker ps -q -f name=${CONTAINER_NAME})" ]; then
#   echo "Stopping existing container..."
#   docker stop "${CONTAINER_NAME}"
#   echo "Removing existing container..."
#   sudo docker rm "${CONTAINER_NAME}"
#   echo "Container removed..."
# fi

# Run a new container from the built image
sudo docker run -d --name "${CONTAINER_NAME}" "${IMAGE_NAME}:${VERSION}"

# Print the result
echo "Docker container '${CONTAINER_NAME}' has been deployed successfully."
