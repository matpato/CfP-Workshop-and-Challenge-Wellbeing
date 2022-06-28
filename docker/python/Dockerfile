# syntax=docker/dockerfile:1

FROM python:3.8-buster

RUN apt update && apt install sudo nano ssh tree iputils-ping net-tools openssh-server bsdmainutils -y

#
# Configure SSH server to allow users to set environment variables
#
RUN sed -i 's/#PermitUserEnvironment no/PermitUserEnvironment yes/' /etc/ssh/sshd_config

WORKDIR /app
RUN mkdir /app/results

#
# Set SSH as a service and start SSH server on port 22
#
RUN service ssh start
CMD [ "/usr/sbin/sshd", "-D" ]
EXPOSE 22

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt

COPY ./app/ .

CMD [ "python3", "./main.py"]
