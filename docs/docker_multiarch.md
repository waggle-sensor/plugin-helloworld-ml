#### Dockerfile for Multi-architecture Support

Docker natively supports multi-architecture through `qemu` library or through `buildx` tool (See [more](https://docs.docker.com/docker-for-mac/multi-arch/)). This means that having a single Dockerfile can be used to build images that support different architecture. SAGE/Waggle platform encourages users to utilize this feature to run their application on various edge devices. With the native support from Docker on this, it is highly recommended to use a single Dockerfile for multi-architecture support. An example of multi-arch support would be,

```
$ cat <<EOF > Dockerfile
FROM ubuntu:18.04
CMD ["/bin/echo", "hello world"]
EOF
# Docker build to make an image for armv7l, arm64, and amd64 (i.e., x86_64)
$ docker buildx build \
  -t helloworld \
  --platform linux/arm/v7,linux/arm64,linux/amd64 \
  --push .
# Run the following command on any of armv7l, arm64, and amd64
$ docker run -ti --rm helloworld
```

However, if a single Dockerfile cannot work on all the target architectures due to missing packages in a particular architecture and etc, it is advised to name Dockerfile as Dockerfile.${arch}. For example,

```
Dockerfile             # Doekerfile for all target architecture
Dockerfile.x86_64      # Dockerfile for Intel x64
Dockerfile.armv7l      # Dockerfile for ARM v7
Dockerfile.arm64_tegra # Dockerfile for ARM64 Nvidia Jetson devices
```
