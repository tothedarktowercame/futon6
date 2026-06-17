#!/usr/bin/env bash
# <UDF name="install_cuda_toolkit" label="Install Ubuntu nvidia-cuda-toolkit?" oneOf="yes,no" default="yes">
# <UDF name="install_linode_cli" label="Install linode-cli via pipx?" oneOf="yes,no" default="yes">
#
# mark4 - Ubuntu 24.04 GPU bootstrap for Linode StackScripts.
# Runs at provision time, installs NVIDIA drivers before the first operator login,
# and reboots so nvidia-smi is available for scripts/linode-4gpu-setup.sh.
set -euo pipefail

LOG=/var/log/mark4-gpu-bootstrap.log
exec > >(tee -a "$LOG") 2>&1

echo "== mark4 GPU bootstrap $(date -Is) =="
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y \
  curl git pipx python3 python3-pip python3-venv \
  linux-headers-"$(uname -r)" ubuntu-drivers-common

if [ "${INSTALL_LINODE_CLI:-yes}" = "yes" ]; then
  echo "== install linode-cli via pipx =="
  pipx ensurepath || true
  pipx install linode-cli || pipx upgrade linode-cli || true
fi

echo "== install recommended NVIDIA driver =="
ubuntu-drivers devices || true
ubuntu-drivers autoinstall

if [ "${INSTALL_CUDA_TOOLKIT:-yes}" = "yes" ]; then
  echo "== install Ubuntu CUDA toolkit package =="
  apt-get install -y nvidia-cuda-toolkit || {
    echo "WARNING: nvidia-cuda-toolkit install failed; driver install still completed."
  }
fi

echo "== installed package summary =="
dpkg -l | grep -E 'nvidia-driver|nvidia-cuda-toolkit|linux-headers' || true

echo "== rebooting so NVIDIA modules bind cleanly =="
touch /var/lib/mark4-gpu-bootstrap-done
systemctl reboot
