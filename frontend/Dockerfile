# frontend/Dockerfile
FROM node:16-alpine as build

WORKDIR /app

# Copy package.json
COPY package.json ./
# Install dependencies
RUN npm install

# Copy the rest of the application
COPY . .

# Build for production
RUN npm run build

# Production environment
FROM nginx:stable-alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80

# Copy the default nginx.conf provided by the image
RUN rm /etc/nginx/conf.d/default.conf
# Add our own 
COPY nginx.conf /etc/nginx/conf.d

CMD ["nginx", "-g", "daemon off;"]
