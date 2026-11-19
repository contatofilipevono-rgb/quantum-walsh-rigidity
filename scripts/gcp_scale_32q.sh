#!/usr/bin/env bash
# Launch a 32-vCPU / 128 GB RAM Google Cloud VM (c2-standard-30 or n2-standard-32),
# compile and run the OpenMP 28-qubit and 32-qubit certified modular flatness benchmark,
# then delete the VM automatically to preserve credits.
set -euo pipefail

ZONE="us-central1-a"
VM_NAME="walsh-rigidity-32cpu"
MACHINE="n2-standard-32" # 32 vCPU, 128 GB RAM (~$1.55/hr -> ~$0.05 for a 2-min run)

gcloud compute instances create "$VM_NAME" \
  --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=20GB

echo "Waiting for SSH..."
sleep 20

gcloud compute scp --zone="$ZONE" src/fwht_omp.c "$VM_NAME":~/fwht_omp.c
gcloud compute ssh --zone="$ZONE" "$VM_NAME" --command="
  sudo apt-get update -qq && sudo apt-get install -y -qq gcc
  gcc -O3 -march=native -fopenmp ~/fwht_omp.c -o ~/fwht_omp
  ~/fwht_omp 24
  ~/fwht_omp 28
  ~/fwht_omp 32
"

echo "Cleaning up VM to save GCP credits..."
gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
