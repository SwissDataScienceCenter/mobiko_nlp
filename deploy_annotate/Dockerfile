# Dockerfile
FROM python:3.11-slim

# --- system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      nginx supervisor apache2-utils curl tini \
    && rm -rf /var/lib/apt/lists/*

# --- dirs & users
ARG BASIC_AUTH_USER=anisia
ARG BASIC_AUTH_PASS_HASH
RUN mkdir -p /mydata/mobiko/anisia/deploy_annotate \
 && printf "%s:%s\n" "$BASIC_AUTH_USER" "$BASIC_AUTH_PASS_HASH" > /mydata/mobiko/anisia/deploy_annotate/htpasswd \
 && chmod 600 /mydata/mobiko/anisia/deploy_annotate/htpasswd

ENV APP_DIR=/app
WORKDIR $APP_DIR

RUN useradd -m -s /bin/bash appuser && mkdir -p $APP_DIR /var/log/supervisor

COPY requirements.txt /app/
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY streamlit_app_annotate.py /app/

RUN rm -f /etc/nginx/conf.d/default.conf

COPY nginx.conf /etc/nginx/conf.d/streamlit.conf
COPY nginx_main.conf /etc/nginx/nginx.conf

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY start.sh /start.sh
RUN chmod +x /start.sh


# --- permissions
RUN chown -R appuser:appuser $APP_DIR /var/log/nginx /var/lib/nginx /var/log/supervisor

USER appuser

# Streamlit listens on 8502 internally, nginx on 8000
ENV STREAMLIT_SERVER_PORT=8502 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 80
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/start.sh"]
