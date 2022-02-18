FROM python:3.10.0

## The MAINTAINER instruction sets the author field of the generated images.
MAINTAINER anna.mccann@epfl.ch

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
