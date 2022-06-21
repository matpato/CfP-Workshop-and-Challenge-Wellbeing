#!/bin/bash
sudo docker run --log-driver local --name py-docker python-docker
sudo docker start py-docker

mkdir -p ./app/logs
sudo docker logs py-docker > ./app/logs/docker_"$(date +"%F_%T")".log  