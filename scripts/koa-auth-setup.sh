#!/usr/bin/env bash
# koa-auth-setup.sh — One-time setup for KOA SSH multiplexing + daily auto-auth.
#
# What it does:
#   1. Bumps ControlPersist to 25h in ~/.ssh/config (if currently lower)
#   2. Installs a macOS launchd agent for daily noon authentication
#   3. Runs an initial authentication
#
# Usage:  ./koa-auth-setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AUTH_SCRIPT="$SCRIPT_DIR/koa-auth.sh"
PLIST_SRC="$SCRIPT_DIR/com.koa.daily-auth.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.koa.daily-auth.plist"
LOG_FILE="$HOME/Library/Logs/koa-auth.log"
SSH_CONFIG="$HOME/.ssh/config"
LABEL="com.koa.daily-auth"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARN:\033[0m %s\n' "$*"; }

# -------------------------------------------------------------------
# 1. Check / update ~/.ssh/config
# -------------------------------------------------------------------
log "Checking SSH config..."

if ! grep -q 'Host.*koa' "$SSH_CONFIG" 2>/dev/null; then
    warn "No 'Host koa' block found in $SSH_CONFIG."
    warn "Add one using the reference at: $SCRIPT_DIR/koa-ssh-config-snippet"
    exit 1
fi

if ! grep -q 'ControlMaster' "$SSH_CONFIG"; then
    warn "No ControlMaster found in SSH config."
    warn "Add multiplexing settings from: $SCRIPT_DIR/koa-ssh-config-snippet"
    exit 1
fi

PERSIST=$(grep 'ControlPersist' "$SSH_CONFIG" | head -1 | awk '{print $2}')
case "$PERSIST" in
    yes|25h)
        log "SSH config has ControlPersist $PERSIST. Good."
        ;;
    "")
        warn "No ControlPersist found. Add 'ControlPersist 25h' to your koa Host block."
        warn "Reference: $SCRIPT_DIR/koa-ssh-config-snippet"
        exit 1
        ;;
    *)
        log "Updating ControlPersist from '$PERSIST' to '25h'..."
        sed -i '' "s/ControlPersist $PERSIST/ControlPersist 25h/" "$SSH_CONFIG"
        log "Updated $SSH_CONFIG"
        ;;
esac

# -------------------------------------------------------------------
# 2. Install launchd agent
# -------------------------------------------------------------------
log "Installing launchd agent..."

chmod +x "$AUTH_SCRIPT"

mkdir -p "$HOME/Library/LaunchAgents"
mkdir -p "$(dirname "$LOG_FILE")"

# Generate plist with resolved paths
sed \
    -e "s|PLACEHOLDER_SCRIPT_PATH|$AUTH_SCRIPT|g" \
    -e "s|PLACEHOLDER_LOG_PATH|$LOG_FILE|g" \
    "$PLIST_SRC" > "$PLIST_DEST"

# Unload if already loaded (ignore errors)
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true

launchctl bootstrap "gui/$(id -u)" "$PLIST_DEST"
log "Launchd agent installed and loaded."
log "It will run daily at 12:00 PM. Logs: $LOG_FILE"

# -------------------------------------------------------------------
# 3. Run initial authentication
# -------------------------------------------------------------------
log "Running initial authentication..."
"$AUTH_SCRIPT"

log ""
log "Setup complete! You should only need to approve DUO once per day."
log ""
log "Useful commands:"
log "  ssh -O check koa              — check if master is alive"
log "  ssh -O exit koa               — kill the master connection"
log "  launchctl list | grep koa     — verify the daily agent"
log "  cat $LOG_FILE  — view auth logs"
