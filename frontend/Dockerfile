FROM nginx:alpine

# Copy our custom Nginx routing and logging configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy the frontend HTML to nginx serve directory
COPY index.html /usr/share/nginx/html/index.html

# Expose port 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]