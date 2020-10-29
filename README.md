# Federated Learning PoC
Server and clients of a Federated Learning implementation

## Quick start

### Installation
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
  
  
## Running the project   
### Server
That's very simple, just go to `federated-learning/src/server` and execute:

    flask run
    
It'll start a master node in http://localhost:5000. To see that's running well, open a browser and go to that URL.
You'll see a message like this:

    Federated Learning server running. Status: IDLE
    
### Clients
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

This is a very early version so it has room for lots of improvements a lot, and probably it will be.

