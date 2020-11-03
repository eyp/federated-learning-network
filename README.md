# Federated Learning PoC
Server and clients of a Federated Learning implementation

## Quick start

There are two options for trying this project, using Docker to create containers for the server and the clients, or
using a standard local installation from the command line.

### Docker installation

Create a network for the server and clients:

    docker network create --driver bridge fl-network

Create the Docker image of the server:

    docker build -t fl-server -f Dockerfile .
    
Run the server:

    docker run --rm --name fl-server -p 5000:5000 --expose=5000 -e FLASK_RUN_HOST=0.0.0.0 -e FLASK_RUN_PORT=5000 --network fl-network fl-server:latest

This command will delete the server container after stopping it. It runs the server on port 5000.

Now, for the client, the first step is creating the Docker image:

    docker build -t fl-client -f Dockerfile .
    
Then, for starting a client running on port 5001 (you can use whatever free port you want):

    docker run --rm --name fl-client-5001 -p 5001:5001 -e CLIENT_URL='http://fl-client-5001:5001' -e SERVER_URL='http://fl-server:5000' -e FLASK_RUN_HOST=0.0.0.0 -e FLASK_RUN_PORT=5001  --network=fl-network fl-client:latest
    
For other clients, simply change the port. For example, for running four clients, do:

    docker run --rm --name fl-client-5002 -p 5002:5002 -e CLIENT_URL='http://fl-client-5002:5002' -e SERVER_URL='http://fl-server:5000' -e FLASK_RUN_HOST=0.0.0.0 -e FLASK_RUN_PORT=5002  --network=fl-network fl-client:latest
    docker run --rm --name fl-client-5003 -p 5003:5003 -e CLIENT_URL='http://fl-client-5003:5003' -e SERVER_URL='http://fl-server:5000' -e FLASK_RUN_HOST=0.0.0.0 -e FLASK_RUN_PORT=5003  --network=fl-network fl-client:latest
    docker run --rm --name fl-client-5004 -p 5004:5004 -e CLIENT_URL='http://fl-client-5004:5004' -e SERVER_URL='http://fl-server:5000' -e FLASK_RUN_HOST=0.0.0.0 -e FLASK_RUN_PORT=5004  --network=fl-network fl-client:latest
    
    
### Command line
If Docker is not an option, then you must install everything and running from the command line.
Python version must be 3.8, I haven't tested it with 3.9 or <3.8 versions.

The best way is to have an isolated environment using conda or similar environment managers.
If you use miniconda or conda, just do:

    conda create --name fedlearning python=3.8
    conda activate fedlearning

Once you're ready to install packages, do this:

     pip install fastai==2.0.16 torch==1.6 torchvision==0.7
     pip install python-dotenv
     pip install aiohttp[speedups]
     pip install flask
    
#### Running the project   
##### Server
That's very simple, just go to `federated-learning/src/server` and execute:

    flask run
    
It'll start a master node in http://localhost:5000. To see that's running well, open a browser and go to that URL.
You'll see a message like this:

    Federated Learning server running. Status: IDLE
    
##### Clients
Open a new console, or just do it in another computer which has access to the server.
Go to `federated-learning/src/client` and execute:

    export CLIENT_URL='http://localhost:5001'
    flask run --port 5001
    
Do that for every client, changing the listening port. You'll see some log traces telling that the client 
has started and has registered in the server:

    Registering in server: http://127.0.0.1:5000
    Doing request http://127.0.0.1:5000/client
    Response received from registration: <Response [201]>
    Client registered successfully
    
### Training sessions
Once we have server and clients running properly and registered, just go to the browser to http://localhost:5000/training.
This will launch a training session between all the clients registered. You can see the progress of the trainings in each 
client's console.

You can do more training sessions afterwards and see how the model improves. If the clients didn't finish its trainings, 
the server will show a message in the console telling that it's still in status CLIENTS_TRAINING, and the new 
training session must wait.

## Customization
You can change some training parameters (epochs, batch size and learning rate) at:

      federated-learning/src/server/server.py line 19
      
Also is possible to increase the samples used for training by the clients at:

      federated-learning/src/client/client.py lines 69,70 for the training dataset and 80,81 for the validation dataset.
      
## Known issues
This is a very first approach, so when clients crash because some bug neither the server or the clients can recover, so 
you must restart everything.

There's no persistence implemented yet, so everytime you start servers & clients the model will be initialized with 
random values and must be trained from the beginning.

This is a very early version, so it has room for lots of improvements, so new features will be added.

