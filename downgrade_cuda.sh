#!/bin/bash

# Remove existing CUDA and NVIDIA packages
echo "Removing existing CUDA and NVIDIA packages..."
sudo apt-get -y --purge remove cuda nvidia-* libnvidia-*
sudo apt-get autoremove

# Add CUDA 11.8 repository
echo "Adding CUDA 11.8 repository..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-D95DBBE2-keyring.gpg /usr/share/keyrings/
sudo apt-get update

# Install CUDA 11.8 toolkit and cuDNN library
echo "Installing CUDA 11.8 toolkit and cuDNN library..."
sudo apt-get -y install cuda=11.8.3-1

# Set environment variables
echo "Setting environment variables..."
echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
echo "Verifying CUDA installation..."
nvcc --version
nvidia-smi

echo "CUDA 11.8 downgrade completed successfully."
