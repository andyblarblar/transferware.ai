FROM node as build

# Copy webapp
COPY static_site static_site
WORKDIR static_site/transferware-app

# Download deps
RUN npm install

# Build app
RUN npm run build

FROM nginx
COPY --from=build /static_site/transferware-app/build /usr/share/nginx/html
EXPOSE 8080