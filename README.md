# Federated Learning Network
Provides a central node and client node implementations to
build a Federated Learning network.

## Quick start

There are two options for trying this project, using Docker to create containers for the server and the clients, or
using a standard local installation from the command line.

### Datasets
For now there are two models for training: MNIST and Chest X-Ray. For using the MNIST one you don't need to install anything else because
the client node downloads the dataset when it runs the training, but for the Chest X-Ray model you'll need a dataset to get it working.

Download the dataset from Kaggle, at https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/download, and uncompress it
wherever you want on the client node machine, in a folder called `chest_xray`. 
By default, the client node looks for it at GLOBAL_DATASETS/chest_xray. The variable GLOBAL_DATASETS is defined 
in the configuration file `client/config.py`.

If you're going to run the client node using Docker, you must pass a volume as a container parameter to indicate where you 
have the datasets:

    -v /your_datasets_directory:/federated-learning-network/datasets

In particular, for Chest X-Ray training, It'll expect a directory `chest_xray` in your datasets directory with at least 
two folders `train`and `test` with x-ray images. 

### Docker installation

Create the Docker image of the server:
    
    cd server
    docker build -t fl-server -f Dockerfile .
    
Run the server:

    docker run --rm --name fl-server -p 5000:5000 fl-server:latest

This command will delete the server container after stopping it. It runs the server on port 5000.

For the client, the first step is creating the Docker image:

    cd client
    docker build -t fl-client -f Dockerfile .
    
#### Running the project   
Now there can be two different scenarios: running nodes on the same IP address, or running each node on a different IP address.
Bear always in mind than we can choose the ports we want if they are free. The ports used in these examples are just that, examples.

##### Same machine
If our IP address is for example 192.168.1.20, and we have the server running on port 5000, we can run several Docker clients in different ports:

    docker run --rm --name fl-client-5001 -p 5001:5000 -e CLIENT_URL='http://192.168.1.20:5001' -e SERVER_URL='http://192.168.1.20:5000' -v /your_datasets_directory:/federated-learning-network/datasets fl-client:latest
    docker run --rm --name fl-client-5002 -p 5002:5000 -e CLIENT_URL='http://192.168.1.20:5002' -e SERVER_URL='http://192.168.1.20:5000' -v /your_datasets_directory:/federated-learning-network/datasets fl-client:latest
    docker run --rm --name fl-client-5003 -p 5003:5000 -e CLIENT_URL='http://192.168.1.20:5003' -e SERVER_URL='http://192.168.1.20:5000' -v /your_datasets_directory:/federated-learning-network/datasets fl-client:latest
    docker run --rm --name fl-client-5004 -p 5004:5000 -e CLIENT_URL='http://192.168.1.20:5004' -e SERVER_URL='http://192.168.1.20:5000' -v /your_datasets_directory:/federated-learning-network/datasets fl-client:latest

If the server is running on another IP address, simply change the variable SERVER_URL accordingly.

**IMPORTANT**: To be able to use the Chest X-Ray model training follow the instructions of _Training the Chest X-Ray model_ section.

##### Every node on a different IP address
If the IP address of the server is, for instance, at 192.168.1.100, and every client will be running on different IP addresses, we can do: 

    docker run --rm --name fl-client -p 5000:5000 -e CLIENT_URL='http://192.168.1.28:5000' -e SERVER_URL='http://192.168.1.100:5000' -v /your_datasets_directory:/federated-learning-network/datasets fl-client:latest
    
For other clients, simply use the right IP address of each one:

    docker run --rm --name fl-client -p 5000:5000 -e CLIENT_URL='http://192.168.1.50:5000' -e SERVER_URL='http://192.168.1.100:5000' -v /your_datasets_directory:/federated-learning-network/datasets fl-client:latest
    docker run --rm --name fl-client -p 5000:5000 -e CLIENT_URL='http://192.168.1.60:5000' -e SERVER_URL='http://192.168.1.100:5000' -v /your_datasets_directory:/federated-learning-network/datasets fl-client:latest
    docker run --rm --name fl-client -p 5000:5000 -e CLIENT_URL='http://192.168.1.70:5000' -e SERVER_URL='http://192.168.1.100:5000' -v /your_datasets_directory:/federated-learning-network/datasets fl-client:latest
    docker run --rm --name fl-client -p 5000:5000 -e CLIENT_URL='http://192.168.1.80:5000' -e SERVER_URL='http://192.168.1.100:5000' -v /your_datasets_directory:/federated-learning-network/datasets fl-client:latest    
    
### Command line
If Docker is not an option, then you must install everything and running from the command line.
Python version must be 3.8, I haven't tested it with 3.9 or <3.8 versions.

The best way is to have an isolated environment using conda or similar environment managers.
If you use miniconda or conda, just do:

    conda create --name fedlearning python=3.8
    conda activate fedlearning

Once you're ready to install packages, do this:

     pip install torch torchvision
     pip install tensorflow
     pip install fastai
     pip install python-dotenv
     pip install aiohttp[speedups]
     pip install flask
    
#### Running the project   
##### Server
That's very simple, just go to `federated-learning-network/server` and execute:

    flask run
    
It'll start a master node in `http://localhost:5000`. To see that's running well, open a browser and go to that URL.
You'll see a message like this:

    Federated Learning server running. Status: IDLE
    
##### Clients
Open a new console, or just do it in another computer which has access to the server.
Go to `federated-learning-network/client` and execute:

    export CLIENT_URL='http://localhost:5001'
    flask run --port 5001
    
Do that for every client, changing the listening port. You'll see some log traces telling the client 
has started and has registered in the server:

    Registering in server: http://127.0.0.1:5000
    Doing request http://127.0.0.1:5000/client
    Response received from registration: <Response [201]>
    Client registered successfully
    
### Training sessions
Once we have server and clients running properly and registered, just open the server dashboard at `http://localhost:5000`
(or use the correct IP address where the server is running). And click on the _Launch training_ button.
This will launch a training session between all the clients registered. You can see the progress of the training in each 
client's console. For example, for MNIST training you will see something like this in the client node console:

    Federated Learning config:
    --Learning Rate: 1.0
    --Epochs: 20
    --Batch size: 256
    
    Training started...
    Accuracy of model trained at epoch 1 : 0.9118
    Accuracy of model trained at epoch 2 : 0.9118
    Accuracy of model trained at epoch 3 : 0.9118
    Accuracy of model trained at epoch 4 : 0.9118
    Accuracy of model trained at epoch 5 : 0.8824
    Accuracy of model trained at epoch 6 : 0.8824
    Accuracy of model trained at epoch 7 : 0.9118
    Accuracy of model trained at epoch 8 : 0.9118
    Accuracy of model trained at epoch 9 : 0.9118
    Accuracy of model trained at epoch 10 : 0.9118
    Accuracy of model trained at epoch 11 : 0.9118
    Accuracy of model trained at epoch 12 : 0.9118
    Accuracy of model trained at epoch 13 : 0.9118
    Accuracy of model trained at epoch 14 : 0.9118
    Accuracy of model trained at epoch 15 : 0.9118
    Accuracy of model trained at epoch 16 : 0.9412
    Accuracy of model trained at epoch 17 : 0.9412
    Accuracy of model trained at epoch 18 : 0.9412
    Accuracy of model trained at epoch 19 : 0.9412
    Accuracy of model trained at epoch 20 : 0.9412
    Training finished...

You can do more training sessions afterwards and see how the model improves. If the clients didn't finish its trainings, 
the server will show a message in the console telling it's still in status CLIENTS_TRAINING, and the new 
training session must wait.

## Customization
You can change some training parameters (epochs, batch size and learning rate) at:

      federated-learning-network/server/server.py start_training method
      
## Known issues
There's no persistence implemented yet, so everytime you start servers & clients the model will be initialized with 
random values and must be trained from the beginning.

This is a very early version, so it has room for lots of improvements, so new features will be added.

