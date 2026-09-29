#!/usr/bin/env bash
# Deploys or redeploys liturgos-auditor (the auditor-stt speech-to-text service) on this server:
#   clone (first run) or pull, create .env.server with a fresh API key (first run), build the image,
#   (re)start the service on 127.0.0.1:4107 for the host's nginx. No database, no migrations: jobs and
#   models live in Docker volumes. The model downloads on the first start (health is 503 until then).
#
#   bash /home/deploy/deploy/liturgos-auditor/scripts/server-deploy.sh
#   (or a copy of this file from anywhere: it clones the repository on the first run)
#
# Environment (all optional):
#   BASE_DIR       /home/deploy/deploy        the checkout goes to $BASE_DIR/liturgos-auditor
#   DEPLOY_BRANCH  main                       branch to deploy
#   REPO_URL       https://github.com/jsilvanus/liturgos-auditor.git
#   DOMAIN         asked on the first run     public domain (only used when .env.server is created)
#   HOST_PORT      4107                       127.0.0.1 port nginx proxies to (first run only)
#
# Settings live in $BASE_DIR/liturgos-auditor/.env.server (mode 600, gitignored). It is never
# overwritten; edit it (model, device, limits) and run this script again to apply changes.
set -euo pipefail

NAME=liturgos-auditor
DEFAULT_DOMAIN=auditor.italeino.fi
BASE_DIR=${BASE_DIR:-/home/deploy/deploy}
APP_DIR=$BASE_DIR/$NAME
BRANCH=${DEPLOY_BRANCH:-main}
REPO_URL=${REPO_URL:-https://github.com/jsilvanus/$NAME.git}
ENV_FILE=$APP_DIR/.env.server
KEPT_ENV=$BASE_DIR/.kept/$NAME.env
SHARED_NETWORK=deploy-shared
SELF=scripts/server-deploy.sh

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }
secret() { openssl rand -hex 32; }

command -v git >/dev/null || die "git is not installed"
command -v openssl >/dev/null || die "openssl is not installed"
docker compose version >/dev/null 2>&1 || die "docker compose (v2) is not available for $(id -un)"

# --- 1. Code ------------------------------------------------------------------------------------
# After updating the checkout, re-run the checkout's own copy of this script (once), so a changed
# script takes effect in the same deployment.
if [ -z "${SERVER_DEPLOY_REEXEC:-}" ]; then
  if [ -d "$APP_DIR/.git" ]; then
    log "Updating $APP_DIR ($BRANCH)"
    git -C "$APP_DIR" fetch --prune origin "$BRANCH"
    git -C "$APP_DIR" checkout -q "$BRANCH"
    git -C "$APP_DIR" pull --ff-only origin "$BRANCH"
  else
    [ -e "$APP_DIR" ] && die "$APP_DIR exists but is not a git checkout"
    log "Cloning $REPO_URL ($BRANCH) into $APP_DIR"
    mkdir -p "$BASE_DIR"
    git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
  fi
  if [ -f "$APP_DIR/$SELF" ]; then
    SERVER_DEPLOY_REEXEC=1 exec bash "$APP_DIR/$SELF" "$@"
  fi
fi
cd "$APP_DIR"
log "Deploying $NAME at commit $(git rev-parse --short HEAD)"

# --- 2. Settings (first run only) ------------------------------------------------------------------
if [ ! -f "$ENV_FILE" ] && [ -f "$KEPT_ENV" ]; then
  log "Restoring settings kept by server-delete.sh ($KEPT_ENV)"
  mv "$KEPT_ENV" "$ENV_FILE"
fi
if [ ! -f "$ENV_FILE" ]; then
  if [ -z "${DOMAIN:-}" ] && [ -t 0 ]; then
    read -r -p "Public domain for $NAME [$DEFAULT_DOMAIN]: " DOMAIN
  fi
  DOMAIN=${DOMAIN:-$DEFAULT_DOMAIN}
  umask 077
  cat > "$ENV_FILE" <<EOF
# liturgos-auditor server settings (scripts/server-deploy.sh). Never commit this file.
# All AUDITOR_STT_* variables are described in README.md ("Environment variables").
DOMAIN=$DOMAIN
HOST_PORT=${HOST_PORT:-4107}

# The service is public through nginx: every endpoint except /health needs
# "Authorization: Bearer <key>". saarnavideo's server script copies this key when it is deployed later.
AUDITOR_STT_API_KEY=$(secret)

# large-v3-turbo on CPU is slow; base or small are lighter. cuda needs the GPU image (docker-compose.gpu.yml).
AUDITOR_STT_MODEL=large-v3-turbo
AUDITOR_STT_DEVICE=cpu
AUDITOR_STT_DEFAULT_LANGUAGE=fi
AUDITOR_STT_JOB_TTL_HOURS=72
AUDITOR_STT_MAX_UPLOAD_MB=2048
AUDITOR_STT_MAX_LIVE_UPLOAD_MB=64
AUDITOR_STT_MAX_QUEUE=8
AUDITOR_STT_CARRY_CONTEXT_CHARS=200
AUDITOR_STT_BATCH_VAD=1
AUDITOR_STT_ALLOWED_MODELS=
EOF
  chmod 600 "$ENV_FILE"
  log "Created $ENV_FILE"
fi

dc() { docker compose -p "$NAME" --project-directory "$APP_DIR" -f docker-compose.server.yml --env-file "$ENV_FILE" "$@"; }
env_value() { sed -n "s/^$1=//p" "$ENV_FILE" | tail -n 1; }

# --- 3. Build and start ----------------------------------------------------------------------------
docker network inspect "$SHARED_NETWORK" >/dev/null 2>&1 || docker network create "$SHARED_NETWORK" >/dev/null

log "Building image"
dc build --pull

log "Starting service"
dc up -d --remove-orphans

# --- 4. Check ------------------------------------------------------------------------------------
# /health is 503 until the model is loaded; the first start downloads it, which can take minutes.
PORT=$(env_value HOST_PORT)
for _ in $(seq 1 30); do
  if curl -fsS -o /dev/null "http://127.0.0.1:$PORT/health"; then
    docker image prune -f >/dev/null
    log "OK: $NAME is up on 127.0.0.1:$PORT (https://$(env_value DOMAIN) once nginx is set up)"
    exit 0
  fi
  sleep 2
done
docker image prune -f >/dev/null
if curl -sS -o /dev/null "http://127.0.0.1:$PORT/health"; then
  log "Started; the model is still loading (first start downloads it). Follow it with:"
  log "  docker compose -p $NAME logs -f auditor-stt"
  exit 0
fi
dc logs --tail 50 auditor-stt
die "$NAME did not answer on http://127.0.0.1:$PORT/health"
