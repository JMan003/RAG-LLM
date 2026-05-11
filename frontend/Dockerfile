FROM nginx:alpine

# Copy the frontend HTML to nginx serve directory
COPY index.html /usr/share/nginx/html/index.html

# Expose port 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
