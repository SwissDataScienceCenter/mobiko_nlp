# Dockerfile
FROM python:3.11-slim

# --- system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      nginx supervisor apache2-utils curl tini \
    && rm -rf /var/lib/apt/lists/*

# --- dirs & users
ENV APP_DIR=/app \
    VENV=/opt/venv
RUN useradd -m -s /bin/bash appuser && mkdir -p $APP_DIR /var/log/supervisor
WORKDIR $APP_DIR

# --- python env
RUN python -m venv $VENV
ENV PATH="$VENV/bin:$PATH"

# --- copy code and configs
# If your app is elsewhere, COPY it in:

COPY streamlit_app_annotate.py $APP_DIR/streamlit_app_annotate.py
COPY requirements.txt $APP_DIR/
COPY nginx.conf /etc/nginx/nginx.conf
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY start.sh /start.sh
RUN chmod +x /start.sh

# --- python deps
RUN pip install --no-cache-dir -r requirements.txt

# --- nginx auth file (mounted at runtime or generated from env)
# We'll expect /etc/nginx/.htpasswd to be provided at runtime,
# or generate it from BASIC_AUTH_USER/PASSWORD in /start.sh

# --- permissions
RUN chown -R appuser:appuser $APP_DIR /var/log/nginx /var/lib/nginx /var/log/supervisor

USER appuser

# Streamlit listens on 8502 internally, nginx on 80
ENV STREAMLIT_SERVER_PORT=8502 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 80
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/start.sh"]
