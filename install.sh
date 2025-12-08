#!/bin/bash

chmod -R 777 /home/ubuntu/ibm_practices/po-conditioning-backend-v1

echo "Deploying .. . ..."
# UV install using wget
wget -qO- https://astral.sh/uv/install.sh | sh
# To add $HOME/.local/bin to your PATH
. $HOME/.local/bin/env

# uv sync to sync dependencies in .venv
uv sync


# add www-data to the group that owns the socket for nginx:
sudo usermod -aG ubuntu www-data
