#!/bin/sh
# Write the basic-auth file from the environment, then run nginx in the
# foreground. Credentials are supplied when the job is submitted, never baked
# into the image: anyone who can pull the image can read what is inside it.
set -eu

if [ -z "${GRAPH_VIEWER_PASSWORD_HASH:-}" ]; then
  echo "GRAPH_VIEWER_PASSWORD_HASH is not set; refusing to serve without a password." >&2
  echo "Generate one with: openssl passwd -apr1" >&2
  exit 1
fi

umask 077
printf '%s:%s\n' "${GRAPH_VIEWER_USER:-mobiko}" "$GRAPH_VIEWER_PASSWORD_HASH" > /tmp/htpasswd

exec nginx -g 'daemon off;'
