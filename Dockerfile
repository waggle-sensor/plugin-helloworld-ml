FROM waggle/plugin-base:1.1.1-ml-cuda11.0-amd64

RUN apt-get update -y
RUN apt-get install -y proj-bin
RUN apt-get install -y libproj-dev
RUN apt-get install -y libgeos-dev
RUN apt-get install -y python3-tk
RUN apt-get install -y python3-nacl

RUN pip3 install --upgrade tensorflow
RUN pip3 install shapely
RUN pip3 install xarray
RUN pip3 install cftime
RUN pip3 install xgboost
RUN pip3 install paramiko
RUN pip3 install highiq

RUN ln -s /usr/local/nvidia /usr/local/cuda-11.0
RUN ln -s /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusolver.so.10 /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusolver.so.11
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.0/targets/x86_64-linux/lib/:${LD_LIBRARY_PATH}"
COPY app/ /app/
COPY app/*.json /app/
COPY *.home_point /app/
WORKDIR /app

ENTRYPOINT ["/usr/bin/python3", "/app/app.py"]
