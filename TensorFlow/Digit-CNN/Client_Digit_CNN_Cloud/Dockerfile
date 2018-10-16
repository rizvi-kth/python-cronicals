FROM nginx
COPY client-app_2/nginx.conf /etc/nginx/nginx.conf
COPY client-app_2 /usr/share/nginx/html
ENV CNN_API=digitcnnapi.westeurope.azurecontainer.io
EXPOSE 8081
#CMD ["nginx", "-g", "daemon off;"]
CMD ["nginx-debug", "-g", "daemon off;"]