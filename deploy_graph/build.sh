#!/usr/bin/env bash
# Build the graph-viewer image from the graphs in output/kg/ (or the ones named).
#
#   deploy_graph/build.sh                         # every output/kg/*/, tag 1.0
#   TAG=1.1 deploy_graph/build.sh output/kg/second_iteration
#   deploy_graph/build.sh --push                  # also push to Docker Hub
#
# Run from anywhere; paths are resolved from the repo root. Rebuild the graphs
# first with `make kg` when the runs change.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
IMAGE="${IMAGE:-docker.io/askinkaty/graph-viewer}"
TAG="${TAG:-1.0}"
PY="${PY:-$REPO/.mobiko-venv/bin/python}"

push=0
graphs=()
for a in "$@"; do
  case "$a" in
    --push) push=1 ;;
    *) graphs+=(--graph "$a") ;;
  esac
done

cd "$REPO"
"$PY" "$HERE/build_site.py" --out "$HERE/site" "${graphs[@]}"
docker build -t "$IMAGE:$TAG" "$HERE"
echo "built $IMAGE:$TAG"

if [ "$push" = 1 ]; then
  docker push "$IMAGE:$TAG"
else
  echo "not pushed; run again with --push, or: docker push $IMAGE:$TAG"
fi
