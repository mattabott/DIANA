#!/bin/bash
# Uninstall the diana-bot service.

set -e

echo "==> stop + disable"
sudo systemctl stop diana-bot 2>/dev/null || true
sudo systemctl disable diana-bot 2>/dev/null || true

echo "==> removing unit file"
sudo rm -f /etc/systemd/system/diana-bot.service

echo "==> daemon-reload"
sudo systemctl daemon-reload

echo "Done. The bot will no longer start at boot and can be run manually."
