FROM waggle/plugin-base-light:0.1.0

COPY app/ /app/
WORKDIR /app

ENTRYPOINT ["/usr/bin/python3", "/app/main.py"]