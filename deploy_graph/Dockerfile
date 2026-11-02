# Static graph viewer: nginx serving deploy_graph/site/ behind basic auth.
# Build with deploy_graph/build.sh, which assembles site/ first.
FROM nginxinc/nginx-unprivileged:1.27-alpine

# Runs as the image's non-root user (uid 101) throughout.
COPY --chown=101:101 nginx.conf /etc/nginx/conf.d/default.conf
COPY --chown=101:101 --chmod=755 start.sh /start.sh
COPY --chown=101:101 site/ /usr/share/nginx/html/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://127.0.0.1:8000/health || exit 1
ENTRYPOINT ["/start.sh"]
