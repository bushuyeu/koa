#!/usr/bin/env bash
# koa-auth.sh — Open (or verify) a persistent SSH master connection to KOA.
#
# The KOA cluster uses password + DUO 2FA (keyboard-interactive).
# This script opens a master connection that persists via ControlPersist.
# All subsequent SSH/koa commands reuse it without any auth prompt.
#
# Modes:
#   Interactive (default): prompts you for password + DUO
#   Automated (--auto):    reads password from macOS Keychain, sends DUO push
#                          via expect. You just tap "Approve" on your phone.
#
# To store your password in Keychain (one-time):
#   security add-generic-password -a pavelb -s koa-ssh -w
#
# Usage:  ./koa-auth.sh [--auto]

set -euo pipefail

HOST="koa"
AUTO=false
if [[ "${1:-}" == "--auto" ]]; then
    AUTO=true
fi

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

# -------------------------------------------------------------------
# 1. If a master connection is already alive, nothing to do.
# -------------------------------------------------------------------
if ssh -O check "$HOST" 2>/dev/null; then
    log "Master connection to $HOST is already active."
    exit 0
fi

log "No active master connection. Opening one now..."

# -------------------------------------------------------------------
# 2. Ensure the SSH key is loaded in the agent.
# -------------------------------------------------------------------
KEY="$HOME/.ssh/id_ed25519_koa"
if [ -f "$KEY" ]; then
    if ! ssh-add -l 2>/dev/null | grep -q "id_ed25519_koa"; then
        log "Loading SSH key into agent..."
        ssh-add --apple-use-keychain "$KEY" 2>/dev/null || ssh-add "$KEY"
    fi
fi

# -------------------------------------------------------------------
# 3a. Automated mode: use expect + Keychain password
# -------------------------------------------------------------------
if $AUTO; then
    log "Auto mode: reading password from macOS Keychain..."

    KOA_PASS=$(security find-generic-password -a pavelb -s koa-ssh -w 2>/dev/null) || {
        log "ERROR: No password found in Keychain."
        log "Store it with: security add-generic-password -a pavelb -s koa-ssh -w"
        exit 1
    }

    if ! command -v expect &>/dev/null; then
        log "ERROR: 'expect' not found (required for --auto mode)."
        exit 1
    fi

    log "Connecting to ${HOST} (auto)... Approve the DUO push on your phone."

    expect <<EXPECT_SCRIPT
set timeout 90
spawn ssh -N -o ControlMaster=yes -o PreferredAuthentications=keyboard-interactive $HOST

# Password prompt
expect {
    -re {[Pp]assword:} {
        send "${KOA_PASS}\r"
    }
    timeout {
        puts "\nERROR: Timed out waiting for password prompt."
        exit 1
    }
}

# DUO prompt — send "1" for push
expect {
    -re {Passcode or option} {
        send "1\r"
    }
    timeout {
        puts "\nERROR: Timed out waiting for DUO prompt."
        exit 1
    }
}

# Wait for DUO approval (the connection stays open with -N)
expect {
    -re {Success} {
        # Auth succeeded, expect will exit and ssh -N keeps running
    }
    -re {Permission denied} {
        puts "\nERROR: Authentication failed."
        exit 1
    }
    timeout {
        puts "\nERROR: Timed out waiting for DUO approval."
        exit 1
    }
}

# Give the master socket a moment to establish
sleep 2
EXPECT_SCRIPT

    # Verify
    sleep 1
    if ssh -O check "$HOST" 2>/dev/null; then
        log "Master connection to $HOST established successfully (auto)."
        exit 0
    else
        log "ERROR: Master connection not found after auto-auth."
        exit 1
    fi
fi

# -------------------------------------------------------------------
# 3b. Interactive mode: user enters password + DUO manually
# -------------------------------------------------------------------
log "Connecting to ${HOST}..."
log "  1. Enter your password"
log "  2. Enter '1' to send a DUO push"
log "  3. Approve on your phone"
echo ""

ssh -N \
    -o ControlMaster=yes \
    -o PreferredAuthentications=keyboard-interactive \
    "$HOST" &
SSH_PID=$!

# Wait for the master socket to appear (auth completes on user's input).
WAITED=0
MAX_WAIT=120
while [ $WAITED -lt $MAX_WAIT ]; do
    if ssh -O check "$HOST" 2>/dev/null; then
        echo ""
        log "Master connection to $HOST established successfully."
        log "All koa commands will now reuse this connection (no more 2FA)."
        exit 0
    fi
    sleep 2
    WAITED=$((WAITED + 2))
done

# If we got here, the connection didn't establish in time.
kill "$SSH_PID" 2>/dev/null || true
log "ERROR: Timed out after ${MAX_WAIT}s waiting for authentication."
exit 1
