# Use a Node base image
FROM node:20

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json before other files
# Utilize Docker cache
COPY src/frontend/package*.json ./

# Install dependencies
RUN npm install
RUN npm install --save vue-good-table

# Copy all files
COPY src/frontend .

# Specify port
EXPOSE 3000

# Run the application
CMD [ "npm", "start" ]
