FROM waggle/plugin-base:1.1.1-ml-cuda10.2-l4t

RUN apt-get update -y
RUN apt-get install -y proj-bin
RUN apt-get install -y libproj-dev
RUN apt-get install -y libgeos-dev
RUN apt-get install -y python3-tk
RUN apt-get install -y python3-nacl

RUN pip3 install numpy==1.17
RUN pip3 install --upgrade --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow
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
