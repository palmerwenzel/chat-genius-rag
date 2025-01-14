#!/bin/bash

# Exit on error
set -e

# Function to check service health
check_health() {
    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
    if [ $response -eq 200 ]; then
        echo "[$(date)] Health check passed"
        return 0
    else
        echo "[$(date)] Health check failed with status $response"
        return 1
    fi
}

# Function to check container status
check_container() {
    container_status=$(docker-compose ps --services --filter "status=running" | grep api || true)
    if [ -n "$container_status" ]; then
        echo "[$(date)] Container is running"
        return 0
    else
        echo "[$(date)] Container is not running"
        return 1
    fi
}

# Function to check system resources
check_resources() {
    echo "[$(date)] System Resources:"
    echo "CPU Usage:"
    top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}'
    echo "Memory Usage:"
    free -m | awk 'NR==2{printf "%.2f%%\n", $3*100/$2}'
    echo "Disk Usage:"
    df -h / | awk 'NR==2{print $5}'
}

# Main monitoring loop
while true; do
    echo "=== Monitoring Check ==="
    
    # Check container status
    if ! check_container; then
        echo "Container down, attempting restart..."
        docker-compose up -d
    fi
    
    # Check service health
    check_health
    
    # Check system resources
    check_resources
    
    echo "======================="
    # Wait for 5 minutes before next check
    sleep 300
done 