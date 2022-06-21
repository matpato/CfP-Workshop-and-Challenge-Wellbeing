#!/bin/bash
sudo systemctl start docker.service
sudo systemctl start containerd.service
docker build --tag python-docker .
