FROM ubuntu:18.04

## System Dependencies
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Python package management and basic dependencies
RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils build-essential
# Register the version in alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
# Set python 3 as the default python
RUN update-alternatives --set python /usr/bin/python3.7

# Upgrade pip to the latest version
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py

# Remove unused files to make the image smaller
RUN apt-get autoremove -yqq --purge \
    && apt-get clean

# Python Packages
COPY requirements.txt /requirements.txt 

# Install Pip Requirements
RUN pip install -r requirements.txt

## Create User Directory
RUN mkdir -p /home/user

ENV PYTHONPATH "${PYTHONPATH}:/home/user"

WORKDIR /home/user