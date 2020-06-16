### Hello World Plugin for Waggle

This repository guides Waggle users to build their own repository containing their application and a manifest file to be running on Waggle nodes. You can use this repository as a template to create your repository (See more on [Github Help](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-from-a-template).

#### Dockerfile

All user applications must be containerized to be running on Waggle nodes or on a [virtual Waggle](https://github.com/waggle-sensor/waggle-node). Dockerfile lets Waggle users build a Docker image containing the application. It is recommended to use [Waggle base images](https://github.com/waggle-sensor/edge-plugins#which-waggle-image-i-choose-for-my-application) for compatibility with the Waggle platform. Waggle platform supports multiple architectures using the same Dockerfile to run the application on various edge devices. If Dockerfile does not work on all the target architecture due to missing packages, it is advised to name Dockerfile as Dockerfile.${arch}. For example,

```
Dockerfile             # Doekerfile for all target architecture
Dockerfile.x86_64      # Dockerfile for Intel x64
Dockerfile.armv7l      # Dockerfile for ARM v7
Dockerfile.arm64_tegra # Dockerfile for ARM64 Nvidia Jetson devices
```

#### Guidance on Developing and Testing

It is beneficial to use command line arguments to feed input(s) and parameters to the main code of your application. This lets the application switch between "dev" and "production" mode easily. This approach makes it very easy to register the applicaiton no our Docker registry through the Edge code repository.

```
# on development get an image from local filesystem
$ python3 app.py -debug -input image.jpg
# on testing get an image from a stream
$ python3 app.py -input http://camera:8090/live
```

#### Registering The Application to Edge Code Repository

The application needs to be registered in the Edge code repository to be running on Waggle nodes. [App specification](app_spec.json) helps to define the application manifest when registering the application. The file can directly be fed into the Edge code repository to register. However, the registration process can also be done via Waggle/SAGE UI and the app_specification.json is not required in this way.
