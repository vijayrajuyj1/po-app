#!/bin/bash

cd /home/ubuntu/ibm_practices/po-conditioning-backend-v1

# Activate virtual environment
source /home/ubuntu/ibm_practices/po-conditioning-backend-v1/.venv/bin/activate

# Run Uvicorn with Unix socket
exec uvicorn main:app --uds /home/ubuntu/ibm_practices/po-conditioning-backend-v1/po_conditioning_backend_app.sock

# exec uvicorn main:app --host 0.0.0.0 --port 8005