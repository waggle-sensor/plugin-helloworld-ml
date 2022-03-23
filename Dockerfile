FROM waggle/plugin-base:1.1.1-ml-cuda11.0-amd64

RUN apt-get update -y
RUN apt-get install -y proj-bin
RUN apt-get install -y libproj-dev
RUN apt-get install -y libgeos-dev
RUN apt-get install -y python3-tk
RUN apt-get install -y python3-nacl

RUN pip3 install shapely
RUN pip3 install xarray
RUN pip3 install cftime
RUN pip3 install xgboost
RUN pip3 install paramiko

COPY app/ /app/
COPY app/*.json /app/
COPY *.home_point /app/
WORKDIR /app

ENTRYPOINT ["/usr/bin/python3", "/app/app.py"]
