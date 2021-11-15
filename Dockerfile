FROM waggle/plugin-base:1.1.1-ml-cuda11.0-amd64

RUN apt-get update -y 
RUN apt-get install -y proj-bin
RUN apt-get install -y libproj-dev
RUN apt-get install -y libgeos-dev
RUN apt-get install -y cuda-11-0
RUN apt-get install -y wget
RUN wget https://www.dropbox.com/s/3z07s2atqgcndxj/sgpdlacfC1.a1.20170731.174445.nc.v0?dl=0 -O sgpdlacfC1.a1.20170731.174445.nc.v0

RUN pip3 install shapely
RUN pip3 install cupy-cuda110
RUN pip3 install act-atmos
RUN pip3 install xarray
RUN pip3 install cftime
RUN pip3 install xgboost
RUN pip3 install highiq

COPY app/ /app/
COPY app/*.json /app/
WORKDIR /app

ENTRYPOINT ["/usr/bin/python3", "/app/app.py"]
