# Use a Node base image
FROM node:14

# Set working directory
WORKDIR /app/src/frontend

# Copy package.json and install dependencies
COPY package.json .
RUN npm install

# Copy rest of the app
COPY . .

# Start the app
CMD ["npm", "start"]