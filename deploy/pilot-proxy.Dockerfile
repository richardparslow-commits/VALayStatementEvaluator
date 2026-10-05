# Dedicated pilot proxy: an explicitly created UID, no writable image layer.
FROM nginx:1.27-alpine
RUN addgroup -g 10001 pilot-proxy && \
    adduser -D -H -s /sbin/nologin -u 10001 -G pilot-proxy pilot-proxy && \
    mkdir -p /run/nginx /run/tls /var/cache/nginx && \
    touch /run/tls/cert.pem /run/tls/key.pem && \
    chown -R 10001:10001 /run/nginx /var/cache/nginx
# Empty TLS mount targets belong to the image, avoiding Docker layer additions.
COPY nginx/pilot.conf /etc/nginx/nginx.conf
USER 10001:10001
# Skip upstream entrypoint scripts that rewrite configuration/runtime files.
ENTRYPOINT ["nginx"]
CMD ["-g", "daemon off;"]
EXPOSE 8443
