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
RUN pip3 install highiq

RUN cd /usr/local/cuda-11.0/lib64
RUN ln -s libcusolver.so.10 libcusolver.so.11
RUN cd /
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.0/lib64;${LD_LIBRARY_PATH}"
RUN echo $LD_LIBRARY_PATH

COPY app/ /app/
COPY app/*.json /app/
COPY *.home_point /app/
WORKDIR /app

ENTRYPOINT ["/usr/bin/python3", "/app/app.py"]
