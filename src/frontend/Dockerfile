# Use a Node base image
FROM node:19

# Set working directory
WORKDIR /app/src/frontend

# Copy package.json and package-lock.json before other files
# Utilize Docker cache
COPY package*.json ./

# Install dependencies
RUN npm install
RUN npm install --save vue-good-table

# Copy all files
COPY . .

# Specify port
EXPOSE 3000

# Run the application
CMD [ "npm", "start" ]