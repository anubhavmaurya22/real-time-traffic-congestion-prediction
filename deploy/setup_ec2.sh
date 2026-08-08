#!/bin/bash
# =============================================================================
# setup_ec2.sh — Run ONCE on a fresh Amazon Linux 2023 EC2 instance via SSH.
#
# Usage (on the EC2 instance after SSH):
#   chmod +x setup_ec2.sh && ./setup_ec2.sh
# =============================================================================
set -euo pipefail

REPO_URL="https://github.com/anubhavmaurya22/real-time-traffic-congestion-prediction.git"
REPO_DIR="/home/ec2-user/streetflow"
SERVICE_NAME="streetflow"

echo "==> [1/6] Updating system packages..."
sudo dnf update -y --quiet

echo "==> [2/6] Installing Python 3.11 + pip + git..."
sudo dnf install -y python3.11 python3.11-pip git --quiet
sudo alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

echo "==> [3/6] Cloning repository..."
if [ -d "$REPO_DIR" ]; then
  echo "    Repo already exists — pulling latest..."
  cd "$REPO_DIR" && git pull
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "==> [4/6] Installing Python dependencies (~5-10 min)..."

# PyTorch CPU-only (inference only, no GPU needed)
pip3.11 install --quiet torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Get the exact version string for PyG wheel URLs
TORCH_VER=$(python3.11 -c "import torch; print(torch.__version__)")
echo "    Installed torch: $TORCH_VER"

# PyG prebuilt wheels (avoids multi-hour source compilation)
pip3.11 install --quiet \
  torch-scatter torch-sparse \
  -f "https://data.pyg.org/whl/torch-${TORCH_VER}+cpu.html"

# Core API dependencies
pip3.11 install --quiet fastapi==0.109.2 uvicorn==0.27.1 networkx==3.2.1 \
  pandas==2.2.0 pydantic==2.6.1

# PyG + temporal
pip3.11 install --quiet torch-geometric==2.5.3
pip3.11 install --quiet torch-geometric-temporal==0.56.2

echo "==> [5/6] Installing systemd service..."
sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null << UNIT
[Unit]
Description=StreetFlow Live - Traffic Congestion Prediction API
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=${REPO_DIR}/api
ExecStart=/usr/bin/python3.11 -m uvicorn api_bangalore:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl start  ${SERVICE_NAME}

echo "==> [6/6] Checking service status..."
sleep 5
sudo systemctl status ${SERVICE_NAME} --no-pager

PUBLIC_IP=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/public-ipv4 || echo "<PUBLIC-IP>")
echo ""
echo "======================================================"
echo "  DONE. API is live at:"
echo "  http://${PUBLIC_IP}:8000"
echo ""
echo "  Quick test:"
echo "  curl http://${PUBLIC_IP}:8000/"
echo ""
echo "  View logs:"
echo "  sudo journalctl -u ${SERVICE_NAME} -f"
echo "======================================================"
