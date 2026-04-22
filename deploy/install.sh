#!/bin/bash
# Install diana-bot as a systemd service.
# Run once, from the project root. Requires sudo.
#
# This script automatically substitutes the <USER> and <PROJECT_DIR>
# placeholders in the .service file with the current system's values,
# then installs it.

set -e
cd "$(dirname "$0")/.."

PROJECT_DIR="$(pwd)"
USER_NAME="${SUDO_USER:-$(whoami)}"

SERVICE_TEMPLATE="deploy/diana-bot.service"
SERVICE_FINAL="/tmp/diana-bot.service.$$"
TARGET="/etc/systemd/system/diana-bot.service"

if [ ! -f "$SERVICE_TEMPLATE" ]; then
    echo "ERROR: $SERVICE_TEMPLATE not found."
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "ERROR: missing .env (copy from .env.example and fill it in)."
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "ERROR: missing venv/ (run: python3 -m venv venv && pip install -r requirements.txt)."
    exit 1
fi

if [ ! -f "config/persona.yaml" ]; then
    echo "ERROR: missing config/persona.yaml (copy from persona.yaml.example)."
    exit 1
fi

echo "==> preparing service file (user=$USER_NAME, dir=$PROJECT_DIR)"
sed "s|<USER>|$USER_NAME|g; s|<PROJECT_DIR>|$PROJECT_DIR|g" \
    "$SERVICE_TEMPLATE" > "$SERVICE_FINAL"

echo "==> copying service file to $TARGET"
sudo cp "$SERVICE_FINAL" "$TARGET"
rm -f "$SERVICE_FINAL"

echo "==> systemctl daemon-reload"
sudo systemctl daemon-reload

echo "==> enable + start"
sudo systemctl enable --now diana-bot

sleep 2
echo ""
echo "==> service status:"
sudo systemctl status diana-bot --no-pager | head -15

echo ""
echo "Useful commands:"
echo "  sudo systemctl status diana-bot       # status"
echo "  sudo journalctl -u diana-bot -f       # live logs"
echo "  sudo journalctl -u diana-bot -n 200   # last 200 lines"
echo "  sudo systemctl restart diana-bot      # restart (after changes)"
echo "  sudo systemctl stop diana-bot         # stop"
echo "  ./deploy/uninstall.sh                 # fully remove"
