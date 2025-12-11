#!/bin/bash
ECR_IMAGE="$1"

sudo docker stop po-api || true
sudo docker rm po-api || true

sudo docker pull $ECR_IMAGE

sudo docker run -d --name po-api \
  -p 3000:8000 \
  -e DB_HOST=$DB_HOST \
  -e DB_PORT=$DB_PORT \
  -e DB_USER=$DB_USER \
  -e DB_PASSWORD=$DB_PASSWORD \
  -e DB_NAME=$DB_NAME \
  --restart=always \
  $ECR_IMAGE
