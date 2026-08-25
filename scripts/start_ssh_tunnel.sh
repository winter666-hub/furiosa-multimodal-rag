#!/usr/bin/env bash
set -euo pipefail

config_file="${1:-.ssh-tunnel.env}"
if [[ ! -f "$config_file" ]]; then
  echo "Missing $config_file. Copy .ssh-tunnel.example to .ssh-tunnel.env first." >&2
  exit 2
fi

# The config file is local-only and must contain shell-compatible KEY=VALUE entries.
# shellcheck disable=SC1090
source "$config_file"

: "${SSH_KEY:?SSH_KEY is required}"
: "${SSH_USER:?SSH_USER is required}"
: "${SSH_HOST:?SSH_HOST is required}"

if [[ ! -f "$SSH_KEY" ]]; then
  echo "SSH key not found: $SSH_KEY" >&2
  exit 2
fi

SSH_PORT="${SSH_PORT:-22}"
LOCAL_LLM_PORT="${LOCAL_LLM_PORT:-8000}"
LOCAL_VISION_PORT="${LOCAL_VISION_PORT:-8001}"
LOCAL_EMBEDDING_PORT="${LOCAL_EMBEDDING_PORT:-8002}"
LOCAL_RERANKER_PORT="${LOCAL_RERANKER_PORT:-8003}"
REMOTE_LLM_PORT="${REMOTE_LLM_PORT:-8000}"
REMOTE_VISION_PORT="${REMOTE_VISION_PORT:-8001}"
REMOTE_EMBEDDING_PORT="${REMOTE_EMBEDDING_PORT:-8002}"
REMOTE_RERANKER_PORT="${REMOTE_RERANKER_PORT:-8003}"

echo "Opening Furiosa API tunnels through ${SSH_USER}@${SSH_HOST}:${SSH_PORT}"
echo "Keep this terminal open; press Ctrl+C to close the tunnels."

exec ssh \
  -i "$SSH_KEY" \
  -p "$SSH_PORT" \
  -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L "127.0.0.1:${LOCAL_LLM_PORT}:127.0.0.1:${REMOTE_LLM_PORT}" \
  -L "127.0.0.1:${LOCAL_VISION_PORT}:127.0.0.1:${REMOTE_VISION_PORT}" \
  -L "127.0.0.1:${LOCAL_EMBEDDING_PORT}:127.0.0.1:${REMOTE_EMBEDDING_PORT}" \
  -L "127.0.0.1:${LOCAL_RERANKER_PORT}:127.0.0.1:${REMOTE_RERANKER_PORT}" \
  "${SSH_USER}@${SSH_HOST}"

