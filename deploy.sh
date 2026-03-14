#!/bin/bash

# ============================================================================
# Provision an AWS EC2 instance and run all ADL experiments on it.
#
# Usage:
#   REPO_URL=<git-clone-url> bash deploy.sh
#
# Optional environment variables:
#   GUROBI_LICENSE  Path to local gurobi.lic file (default: $HOME/gurobi.lic)
#   INSTANCE_TYPE   EC2 instance type          (default: c5.4xlarge)
#   VOLUME_SIZE     Root volume size in GB     (default: 30)
#   KEY_NAME        EC2 key pair name          (default: adl-key)
#   KEY_FILE        Local path for .pem file   (default: adl-key.pem)
#   SG_NAME         Security group name        (default: adl-sg)
# ============================================================================

set -e

REPO_URL="${REPO_URL:?Please set the REPO_URL environment variable}"
GUROBI_LICENSE="${GUROBI_LICENSE:-$HOME/gurobi.lic}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c5.4xlarge}"
VOLUME_SIZE="${VOLUME_SIZE:-30}"
KEY_NAME="${KEY_NAME:-adl-key}"
KEY_FILE="${KEY_FILE:-adl-key.pem}"
SG_NAME="${SG_NAME:-adl-sg}"

box() {
    local msg="$1"
    local len=${#msg}
    local line=$(printf '%*s' "$((len + 2))" | tr ' ' '-')
    echo "+${line}+"
    echo "| ${msg} |"
    echo "+${line}+"
}

# ── Validate Gurobi license ──────────────────────────────────────
if [ ! -f "$GUROBI_LICENSE" ]; then
    box "[ERROR] Gurobi license not found at $GUROBI_LICENSE"
    echo "Set GUROBI_LICENSE to the path of your gurobi.lic file."
    exit 1
fi

# ── 1. Key pair ──────────────────────────────────────────────────
box "[INFO] Setting up key pair..."
if [ ! -f "$KEY_FILE" ]; then
    aws ec2 create-key-pair \
        --key-name $KEY_NAME \
        --query 'KeyMaterial' \
        --output text > $KEY_FILE
    chmod 400 $KEY_FILE
    box "[INFO] Key pair created: $KEY_FILE"
else
    box "[INFO] Key file already exists: $KEY_FILE"
fi

# ── 2. Security group ─────────────────────────────────────────────
box "[INFO] Setting up security group..."
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' \
    --output text 2>/dev/null)

if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
    SG_ID=$(aws ec2 create-security-group \
        --group-name $SG_NAME \
        --description "ADL experiments" \
        --query 'GroupId' --output text)
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp --port 22 --cidr 0.0.0.0/0
    box "[INFO] Security group created: $SG_ID"
else
    box "[INFO] Security group already exists: $SG_ID"
fi

# ── 3. Launch instance ────────────────────────────────────────────
box "[INFO] Looking up latest Amazon Linux 2023 AMI..."
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=al2023-ami-*-x86_64" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

box "[INFO] Launching EC2 instance ($INSTANCE_TYPE, ${VOLUME_SIZE}GB)..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SG_ID \
    --block-device-mappings "[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE}}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

box "[INFO] Waiting for instance $INSTANCE_ID to pass status checks..."
aws ec2 wait instance-status-ok --instance-ids $INSTANCE_ID

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

box "[INFO] Instance ready at $PUBLIC_IP"

# ── 4. Copy Gurobi license ────────────────────────────────────────
box "[INFO] Copying Gurobi license..."
scp -i $KEY_FILE \
    -o StrictHostKeyChecking=no \
    $GUROBI_LICENSE ec2-user@$PUBLIC_IP:~/gurobi.lic

# ── 5. Clone repo and launch experiments in background ───────────
box "[INFO] Deploying and starting experiments..."
ssh -i $KEY_FILE \
    -o StrictHostKeyChecking=no \
    ec2-user@$PUBLIC_IP << EOF
git clone $REPO_URL ApplicationDrivenLearningExperiments.jl
cd ApplicationDrivenLearningExperiments.jl
nohup bash main.sh > ~/experiments.log 2>&1 &
echo \$! > ~/experiments.pid
echo "Experiments started (PID: \$(cat ~/experiments.pid))"
EOF

# ── Summary ───────────────────────────────────────────────────────
box "[SUCCESS] Deployment complete!"
echo ""
echo "  Instance ID  : $INSTANCE_ID"
echo "  Public IP    : $PUBLIC_IP"
echo ""
echo "  Follow logs  : ssh -i $KEY_FILE ec2-user@$PUBLIC_IP 'tail -f ~/experiments.log'"
echo "  Stop instance: aws ec2 stop-instances --instance-ids $INSTANCE_ID"
echo "  Term instance: aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
