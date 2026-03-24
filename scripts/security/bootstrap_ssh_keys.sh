#!/usr/bin/env bash
# scripts/security/bootstrap_ssh_keys.sh
# Distribute SSH public key to Dog & Server machines.
set -euo pipefail
USER_NAME="${USER_NAME:-$(whoami)}"
SSH_PORT="${SSH_PORT:-22}"
KEY_FILE="${KEY_FILE:-$HOME/.ssh/id_ed25519.pub}"

if [ ! -f "$KEY_FILE" ]; then
  echo "SSH public key not found at $KEY_FILE"
  echo "Generate one with: ssh-keygen -t ed25519"
  exit 1
fi

for HOST in 10.0.0.18 10.150.16.29; do
  echo "[*] Distributing key to $USER_NAME@$HOST:$SSH_PORT"
  ssh -p "$SSH_PORT" "$USER_NAME@$HOST" 'mkdir -p ~/.ssh && chmod 700 ~/.ssh'
  cat "$KEY_FILE" | ssh -p "$SSH_PORT" "$USER_NAME@$HOST" 'cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
  echo "    -> done"
done
