#!/usr/bin/env bash
# Usage: scripts/vm.sh start|stop|sync|ssh [cmd]
set -euo pipefail
RG=ROGII-RG; VM=rogii-gpu; HOST=rogii; REMOTE=/mnt/voice/repo
HERE="$(cd "$(dirname "$0")/.." && pwd)"
case "${1:-}" in
  start) az vm start -g $RG -n $VM -o none && echo started ;;
  stop)  az vm deallocate -g $RG -n $VM -o none && echo deallocated ;;
  sync)  rsync -az --delete --exclude .git --exclude __pycache__ --exclude 'data/public' \
           "$HERE/" "$HOST:$REMOTE/" && echo synced ;;
  ssh)   shift; ssh $HOST "cd $REMOTE && export HF_HOME=/mnt/voice/hf && source ~/voice-venv/bin/activate && $*" ;;
  *) echo "usage: $0 start|stop|sync|ssh [cmd]"; exit 1 ;;
esac
