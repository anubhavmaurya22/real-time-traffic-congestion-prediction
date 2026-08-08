#!/bin/bash
# =============================================================================
# launch_ec2.sh  —  Run from YOUR LOCAL MACHINE (Windows Git Bash / WSL).
#
# Prerequisites:
#   1. aws configure  (Access Key, Secret, region=ap-south-1, output=json)
#   2. A key pair named "streetflow-key" already created in AWS Console
#      OR let this script create one.
#
# What it does:
#   - Creates a security group allowing ports 22 (SSH) and 8000 (API)
#   - Launches a t3.small instance with Amazon Linux 2023
#   - Prints the public IP and SSH command
# =============================================================================
set -euo pipefail

REGION="ap-south-1"          # Mumbai — closest to Bangalore
KEY_NAME="streetflow-key"
SG_NAME="streetflow-sg"
INSTANCE_TYPE="t3.small"
# Amazon Linux 2023 AMI for ap-south-1 (update if stale: aws ec2 describe-images ...)
AMI_ID="ami-0f58b397bc5c1f2e8"

echo "==> Creating key pair (saves to ~/.ssh/${KEY_NAME}.pem)..."
if aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" &>/dev/null; then
  echo "    Key pair '${KEY_NAME}' already exists — skipping creation."
else
  aws ec2 create-key-pair \
    --key-name "$KEY_NAME" \
    --region "$REGION" \
    --query "KeyMaterial" \
    --output text > ~/.ssh/${KEY_NAME}.pem
  chmod 400 ~/.ssh/${KEY_NAME}.pem
  echo "    Key saved to ~/.ssh/${KEY_NAME}.pem"
fi

echo "==> Creating security group '${SG_NAME}'..."
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=${SG_NAME}" \
  --region "$REGION" \
  --query "SecurityGroups[0].GroupId" \
  --output text 2>/dev/null || echo "None")

if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
  SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "StreetFlow API — SSH + port 8000" \
    --region "$REGION" \
    --query "GroupId" \
    --output text)
  echo "    Created security group: $SG_ID"

  # Allow SSH from anywhere (restrict to your IP in production)
  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp --port 22 --cidr 0.0.0.0/0 \
    --region "$REGION"

  # Allow API port
  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp --port 8000 --cidr 0.0.0.0/0 \
    --region "$REGION"
else
  echo "    Security group already exists: $SG_ID"
fi

echo "==> Launching EC2 instance (${INSTANCE_TYPE}, ${AMI_ID})..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --count 1 \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --region "$REGION" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=streetflow-api}]" \
  --query "Instances[0].InstanceId" \
  --output text)

echo "    Instance ID: $INSTANCE_ID"
echo "==> Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --region "$REGION" \
  --query "Reservations[0].Instances[0].PublicIpAddress" \
  --output text)

echo ""
echo "======================================================"
echo "  Instance running!"
echo "  Public IP : $PUBLIC_IP"
echo ""
echo "  Next — SSH into the instance and run setup:"
echo ""
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP}"
echo ""
echo "  Then on the server:"
echo "  curl -O https://raw.githubusercontent.com/anubhavmaurya22/real-time-traffic-congestion-prediction/main/deploy/setup_ec2.sh"
echo "  chmod +x setup_ec2.sh && ./setup_ec2.sh"
echo ""
echo "  API will be at: http://${PUBLIC_IP}:8000"
echo "======================================================"
