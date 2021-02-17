FROM waggle/plugin-base:1.0.0-ml-cuda11.0-amd64

RUN apt-get update -y 
RUN apt-get install -y proj-bin
RUN apt-get install -y libproj-dev
RUN apt-get install -y libgeos-dev
RUN apt-get install -y gcc
RUN apt-get install -y g++
RUN apt-get install -y python3-dev
RUN apt-get install -y cuda-11-0

RUN pip3 install --upgrade cython numpy pyshp six
RUN pip3 install shapely
RUN pip3 install act-atmos
RUN pip3 install cupy
RUN pip3 install highiq
RUN pip3 install xgboost
RUN pip3 install cftime

COPY app/ /app/
COPY app/*.json /app/
COPY *.nc /app/
WORKDIR /app

ENTRYPOINT ["/usr/bin/python3", "/app/app.py"]
