#!/bin/bash

# Exit on error
set -e

echo "Starting deployment..."

# Update system packages
sudo yum update -y

# Install Docker if not installed
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo yum install -y docker
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker $USER
    echo "Added $USER to docker group. You may need to log out and back in for this to take effect."
fi

# Install Docker Compose if not installed
if ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Stop and remove existing containers
echo "Cleaning up existing containers..."
docker-compose down || true

# Pull latest changes
echo "Pulling latest changes..."
git pull origin master

# Build and start containers
echo "Building and starting containers..."
docker-compose up --build -d

# Wait for health check
echo "Waiting for service to be healthy..."
attempts=0
max_attempts=30
until $(curl --output /dev/null --silent --fail http://localhost:8000/health); do
    if [ ${attempts} -eq ${max_attempts} ]; then
        echo "Service failed to become healthy"
        exit 1
    fi
    
    printf '.'
    attempts=$(($attempts+1))
    sleep 2
done

echo "Deployment completed successfully!" 