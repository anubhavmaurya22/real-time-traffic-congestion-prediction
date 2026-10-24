#!/bin/bash
# Fast EC2 install — installs via prebuilt wheels only, no compilation needed.
set -e

REPO_URL="https://github.com/anubhavmaurya22/real-time-traffic-congestion-prediction.git"
REPO_DIR="/home/ec2-user/streetflow"
SERVICE_NAME="streetflow"

echo "==> [1/6] Installing pip and build tools..."
python3.11 -m ensurepip --upgrade 2>/dev/null || true
python3.11 -m pip install --upgrade pip --quiet

echo "==> [2/6] Cloning / updating repo..."
if [ -d "$REPO_DIR" ]; then
  cd "$REPO_DIR" && git pull
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "==> [3/6] Installing PyTorch (CPU-only)..."
python3.11 -m pip install --quiet torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Get exact torch version string for PyG wheel URLs
TORCH_VER=$(python3.11 -c "import torch; v=torch.__version__; print(v.split('+')[0]+'%2Bcpu')")
echo "    Torch version string for PyG wheels: $TORCH_VER"

echo "==> [4/6] Installing PyG + torch-geometric-temporal (prebuilt wheels)..."
python3.11 -m pip install --quiet \
  torch-scatter torch-sparse \
  -f "https://data.pyg.org/whl/torch-${TORCH_VER}.html"

python3.11 -m pip install --quiet torch-geometric==2.5.3
python3.11 -m pip install --quiet torch-geometric-temporal==0.56.2

echo "==> [5/6] Installing FastAPI stack + runtime deps..."
python3.11 -m pip install --quiet \
  fastapi==0.109.2 uvicorn==0.27.1 networkx==3.2.1 \
  pandas==2.2.0 pydantic==2.6.1 httpx==0.27.0 \
  pyyaml==6.0.1 joblib==1.3.2 scikit-learn==1.4.1.post1

echo "==> [6/6] Installing systemd service..."

# Generate a random admin reload token if not already set
if [ ! -f /home/ec2-user/.streetflow_env ]; then
  ADMIN_TOKEN=$(python3.11 -c "import secrets; print(secrets.token_urlsafe(32))")
  echo "ADMIN_RELOAD_TOKEN=${ADMIN_TOKEN}" > /home/ec2-user/.streetflow_env
  chmod 600 /home/ec2-user/.streetflow_env
  echo "  Generated admin reload token (stored in /home/ec2-user/.streetflow_env)"
fi

sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null <<UNIT
[Unit]
Description=StreetFlow Live - Traffic Congestion Prediction API
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=${REPO_DIR}/api
EnvironmentFile=/home/ec2-user/.streetflow_env
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
sudo systemctl restart ${SERVICE_NAME}

sleep 5
sudo systemctl status ${SERVICE_NAME} --no-pager

echo ""
echo "======================================================"
echo "  API live at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000"
echo "  Health:      curl http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000/health"
echo "======================================================"
