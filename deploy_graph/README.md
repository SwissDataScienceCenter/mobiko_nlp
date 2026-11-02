# Graph viewer deployment

Serves the knowledge-graph viewer (`src/graph/render_html.py`) as static pages
behind basic auth, on RunAI, the same way `deploy_annotate/` serves the
annotation app. No database: each graph is one self-contained page.

```
/health                     ok, no auth (for the RunAI probe)
/graph/                     index: one card per graph
/graph/<graph>/             the viewer for output/kg/<graph>/
/graph/vendor/...           vis-network, pinned and hash-checked at build time
```

## Build

```bash
make kg RUNS=...                                # (re)build the graph(s) first
deploy_graph/build.sh                           # every output/kg/*/ -> docker.io/askinkaty/graph-viewer:1.0
TAG=1.1 deploy_graph/build.sh output/kg/second_iteration
deploy_graph/build.sh --push                    # also push
```

`build.sh` runs `build_site.py` (renders each graph into `site/`, writes the
index, fetches vis-network) and then `docker build`. The graphs are baked into
the image, so new graphs mean a new tag.

## Run

```bash
openssl passwd -apr1                            # prompts; prints a hash like $apr1$...
runai inference submit graph-viewer -p mobiko-anisia \
  -i docker.io/askinkaty/graph-viewer:1.0 --serving-port=8000 \
  -e GRAPH_VIEWER_USER=mobiko -e 'GRAPH_VIEWER_PASSWORD_HASH=$apr1$...'
```

Keep the single quotes: the hash contains `$`. Locally:

```bash
docker run --rm -p 127.0.0.1:8090:8000 -e 'GRAPH_VIEWER_PASSWORD_HASH=$apr1$...' \
  docker.io/askinkaty/graph-viewer:1.0          # http://localhost:8090/graph/
```

## How it differs from deploy_annotate, and why

* **Credentials at start-up, not build time.** The password hash comes from the
  environment of the job. Anything baked into an image can be read by anyone
  who can pull it, and a public Docker Hub repository means everyone. The
  container refuses to start without `GRAPH_VIEWER_PASSWORD_HASH`.
* **Do not pass `--command` to `runai`.** The image's entrypoint writes the auth
  file and starts nginx; overriding it would serve nothing, rather than
  serve without a password.
* **Non-root nginx** (`nginx-unprivileged`, uid 101), and it also runs under an
  arbitrary uid. Everything it writes goes to `/tmp`.
* **The graph data is in the image.** Basic auth protects the running site, not
  the image: if `docker.io/askinkaty/graph-viewer` is public, the graphs are
  public. Make the repository private if that matters.
