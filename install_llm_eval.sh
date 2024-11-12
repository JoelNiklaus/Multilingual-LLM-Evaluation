#!/bin/bash

activate () {
    echo "Activating virtual environment..."
    . .venv/bin/activate
}

sudo apt update
# sudo apt upgrade -y  # Is this really necessary?
sudo apt install -y python3 python3-pip python3-venv git
sudo apt autoremove -y

python3 -m venv .venv
activate

pip install -r requirements.txt # Directly installing from requirements-freeze.txt doesn't work
# pip install --upgrade cuda-python # Do this if the current cuda version is outdated

# sudo reboot now  # Is this really necessary?