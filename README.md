### Hello World Plugin for Machine Learning in SAGE/Waggle

This repository guides SAGE/Waggle users on how to build their own repository containing their machine learning application, called a plugin, a Dockerfile that describes how to build the application, and a manifest file to specify metadata of the application being able to run it on SAGE/Waggle nodes. You can use this repository as a template to create your repository (See more on [Github Help](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-from-a-template)). The following sections describe steps to build this hello world ML plugin that can be running on SAGE/Waggle nodes.

#### Step 1: Prepare A Development Environment

You can use your laptop (or desktop) to train your model and develop your plugin or use one of the computing nodes SAGE project offers. The support includes high performance computing (HPC) grade computing nodes with and without GPU acceleration. Refer to [Chameleon guide]() to start with Chameleon for your development environment.

One of the easiest ways to build machine learning applications is to use [Jupyter](https://jupyter.org). As of June 2020, we currently support Python3 as a programming language for running plugins. Other programming languagues such as C++ and Go may be supported in the future.

#### Step 2: Prepare Dataset

SAGE offers various types of datasets and is adding more datasets as we collect more samples from the SAGE/Waggle nodes already deployed in the field. In this hello world example, we use [face mask dataset](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset) from Kaggle. Download it on your development environment. You can bring your own dataset as well.

#### Step 3: Training A Model

To spin up an instance of Jupyter iPython notebook,
```
# assume Docker 19.xx.xx or higher is already installed on the environment
# run below on the computing node (e.g., Chameleon node or your laptop)
$ docker run -d \
  --name tf-jupyter \
  --gpus all \     # if the Docker supports Nvidia docker runtime
  -p 8888:8888 \   # expose 8888 port to the host computing node
  -v $(pwd):/tf \  # $(pwd) is the directory where your ipython script locates
  tensorflow/tensorflow:latest-gpu-jupyter
$ docker logs tf-jupyter
...    
    Or copy and paste one of these URLs:
        http://e3b293e033ea:8888/?token=8a5764515a84243e7c0a5dcae9aaacebdfbefbdb72bae0f6
     or http://127.0.0.1:8888/?token=8a5764515a84243e7c0a5dcae9aaacebdfbefbdb72bae0f6

# if the docker runs remotely, run the following
# to open a port from your laptop to the computing node
$ ssh -L 8888:localhost:8888

# replace ${TOKEN} with the actual token you get from the above
$ open http://127.0.0.1:8888/?token=${TOKEN}
```

Follow [the jupyter notebook](docs/training_mask_classifier.ipynb) to train a convolutional neural network (CNN) model to classify people with and without wearing a mask.

#### Step 4: Build A Plugin

It is beneficial to use command line arguments to feed input(s) and parameters to the main code of your application. This lets the application switch between "dev" and "production" mode easily. This approach makes it easy to register the applicaiton to our Docker registry through the Edge code repository as then our ECR uses the command line argument to test and profile the application in order to "certify" it.

```
# on development get an image from local filesystem
$ python3 app.py -debug -input image.jpg
# on testing get an image from a stream of a camera
$ python3 app.py -input http://camera:8090/live
```

#### Step 5: Make Dockerfile

All user applications must be containerized to be running on SAGE/Waggle nodes or on a [virtual Waggle](https://github.com/waggle-sensor/waggle-node). Dockerfile allows users to build a Docker image containing the application with and without models and dependent libraries. It is strictly required to use [Waggle base images](https://github.com/waggle-sensor/edge-plugins#which-waggle-image-i-choose-for-my-application) for the compatibility with the SAGE/Waggle platform. However, there can be an exception that users might want to use other base image (e.g., Nvidia-docker image) to build. That should not be a problem for SAGE/Waggle nodes to run it. However, it may take more time/effort to "certify" that the application can be running on the nodes. The Dockerfile is usually located in the root of the application file structure. The name of Dockerfile should not be changed unless it has to.

If the application needs to be running on multiple architecture platforms (e.g., amd64, arm64, and armv7), please refer to [Dockerfile.arch](docs/docker_multiarch.md). This hello world ML plugin requires Tensorflow Lite runtime library and the name of the library differs for different architecture (as of June 2020). It is therefore necessary to have multiple Dockerfile to support multiple architectures. Please refer to [Dockerfile.armv7](Dockerfile.armv7), [Dockerfile.arm64](Dockerfile.arm64), and [Dockerfile.amd64](Dockerfile.amd64) for detail.

#### Step 6: Registering The Application to Edge Code Repository

The application needs to be registered in the Edge code repository to be running on SAGE/Waggle nodes. [App specification](sage.json) helps to define the application manifest when registering the application. The file can directly be fed into the Edge code repository to register. The registration process can also be done via Waggle/SAGE UI and the app_specification.json is not required in this way.
