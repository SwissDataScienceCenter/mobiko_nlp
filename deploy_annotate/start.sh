#!/usr/bin/env bash
set -euo pipefail
#
#
##!/usr/bin/env bash
#set -euo pipefail
#
## Must match the path used in nginx.conf
#NGINX_AUTH_FILE="${NGINX_AUTH_FILE:-/mydata/mobiko/anisia/deploy_annotate/htpasswd}"
#
#if [ ! -r "$NGINX_AUTH_FILE" ]; then
#  echo "ERROR: Auth file '$NGINX_AUTH_FILE' not readable. Mount it at that exact path (same as nginx.conf)."
#  exit 1
#fi
#
#
## Launch supervisor (which runs nginx + streamlit)
#exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf


# Must match nginx.conf
NGINX_AUTH_FILE="${NGINX_AUTH_FILE:-/mydata/mobiko/anisia/deploy_annotate/htpasswd}"
[ -r "$NGINX_AUTH_FILE" ] || { echo "Auth file not readable: $NGINX_AUTH_FILE"; exit 1; }

# Streamlit on 0.0.0.0:8502 and behind /annotate
STREAMLIT_SERVER_ADDRESS="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"
STREAMLIT_SERVER_PORT="${STREAMLIT_SERVER_PORT:-8502}"
STREAMLIT_BASE_PATH="${STREAMLIT_BASE_PATH:-annotate}"

python -c "import streamlit" 2>/dev/null || python -m pip install --no-cache-dir streamlit

# Start Streamlit in the background
python -m streamlit run /app/streamlit_app_annotate.py \
  --server.address "$STREAMLIT_SERVER_ADDRESS" \
  --server.port "$STREAMLIT_SERVER_PORT" \
  --server.baseUrlPath "$STREAMLIT_BASE_PATH" &

# Nginx in foreground
exec /usr/sbin/nginx -g 'daemon off;'
