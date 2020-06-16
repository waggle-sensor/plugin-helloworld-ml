### Hello World Plugin for Waggle

This repository guides Waggle users to build their own repository containing their application and a manifest file to be running on Waggle nodes. You can use this repository as a template to create your repository (See more on [Github Help](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-from-a-template)).

#### Dockerfile

All user applications must be containerized to be running on Waggle nodes or on a [virtual Waggle](https://github.com/waggle-sensor/waggle-node). Dockerfile lets Waggle users build a Docker image containing the application and dependent libraries. It is strictly required to use [Waggle base images](https://github.com/waggle-sensor/edge-plugins#which-waggle-image-i-choose-for-my-application) for compatibility with the Waggle platform. However, there can be an exception that users might want to use other base image (e.g., Nvidia-docker image) to build. That should not be a problem for Waggle nodes to run it, it may take more time/effort to "certify" that the application can be running on the nodes. The Dockerfile is usually located in the root of the application file structure. The name of Dockerfile should not be changed unless it has to.

#### Dockerfile for Multi-architecture Support

Docker natively supports multi-architecture through `qemu` library or through `buildx` tool (See [more](https://docs.docker.com/docker-for-mac/multi-arch/)). This means that having a single Dockerfile can be used to build images that support different architecture. Waggle platform encourages users to utilize this feature to run their application on various edge devices. With the native support from Docker on this, it is highly recommended to use a single Dockerfile for multi-architecture support. An example of multi-arch support would be,

```
$ cat <<EOF > Dockerfile
FROM ubuntu:18.04
CMD ["/bin/echo", "hello world"]
EOF
# Docker build to make an image for armv7l, arm64, and amd64 (i.e., x86_64)
$ docker buildx \
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

#### Guidance on Developing and Testing

It is beneficial to use command line arguments to feed input(s) and parameters to the main code of your application. This lets the application switch between "dev" and "production" mode easily. This approach makes it very easy to register the applicaiton to our Docker registry through the Edge code repository.

```
# on development get an image from local filesystem
$ python3 app.py -debug -input image.jpg
# on testing get an image from a stream of a camera
$ python3 app.py -input http://camera:8090/live
```

#### Registering The Application to Edge Code Repository

The application needs to be registered in the Edge code repository to be running on Waggle nodes. [App specification](sage.json) helps to define the application manifest when registering the application. The file can directly be fed into the Edge code repository to register. However, the registration process can also be done via Waggle/SAGE UI and the app_specification.json is not required in this way.
