
# Neural Network Implementation    
    
## Introduction 
This project was developed as part of my coursework for the Deep Learning I course during my college studies (Apr/2024). It implements a basic neural network framework in Python, utilizing only Numpy for computation throughout the network training process. The framework offers classes for creating and training neural networks with customizable architectures and activation functions.    
    
## Project Overview The project consists of two main classes:    
- `Network`: Represents the entire neural network structure, including methods for forward pass, backward pass (backpropagation), gradient descent optimization, training, and prediction.    
- `Layer`: Represents a single layer of neurons in the network, including methods for initializing parameters, performing forward pass, and applying activation functions.    
    
    
    
## Computing Process and Hyperparameters 
### Computing Process:      
The neural network framework implemented in this project follows a standard process for training neural networks. Here's an overview of the computing process:      
      
#### Initialization 
Initialize the neural network with specified hyperparameters such as the number of layers, activation functions, and initialization methods for weights and biases.      
      
#### Forward Pass 
Perform a forward pass through the network to compute the predicted output for a given input. This involves computing the weighted sum of inputs and applying activation functions at each layer.      
      
- compute layer's pre-activations:      
$$z_i = \beta_i +  \sum_{j=1}^{D} w_{ij}x_j$$      
- compute layer's output:      
$$a_i = \sigma[z_i]$$      
- or in the vectorial form:      
$${\bf z_i} = {\bf W . x}$$   
$${\bf a} = \sigma [{\bf z_i}]$$     
where $\sigma$ is one of the following activation functions:
1. **'sigmoid'**
$$\sigma(z) = \frac{1}{1+e^{-z}}$$
2. **'relu'**
$$ReLU(z) = max(0, z)$$
3. **'leaky_relu'**
$$LeakyReLU(z) = max(\alpha z, z)\  \ ,   \alpha =0.01$$
4. **'tanh'**
$$tanh(z) = \frac{e^t - e^{-t}}{e^t + e^{-t}}$$
5. **'softmax'**
$$\sigma(z_i) = \frac{e^{z_{i}}}{\sum_{j=1}^K e^{z_{j}}} \ \ \ for\ i=1,2,\dots,K$$
6. **'linear'**
$$Linear(z) = z$$
      
     
#### Loss Computation 
Calculate the loss between the predicted output and the actual output (ground truth). The loss function used depends on the nature of the problem (e.g., binary cross-entropy for binary classification, mean squared error for regression).      
    
- **Loss functions implemented:** 
1. **'mse': mean squared error**: can be used for regression problems. It is derived from the normal distribution. Expects a vector of real numbers as output.    
  
$$L_{MSE} =\frac{1}{N} \sum_{i=1}^N{(y_i - \hat{y}_i)}^2$$    
        
2. **'bce': binary cross-entropy**: can be used for binary classification problems. It is derived from the Bernoulli distribution. Expects an output vector of length 2 where the values are probabilities of belonging to the respective class.  
   
$$L_{BCE} = {-(y\log(\hat{y}) + (1 - y)\log(1 - \hat{y}))}$$   
   
3. **'cce': categorical cross-entropy**: can be used for multiclass classification problems. It is derived from the categorical distribution. Expects an output vector of length *n_classes* where the values are probabilities of belonging to the respective class.    
   
$$L_{CCE} = -\sum_{c=1}^My_{o,c}\log(\hat{y}_{o,c})$$    
                  
#### Backpropagation 
Compute the gradients of the loss function with respect to the network parameters (weights and biases) using backpropagation. This involves propagating the error backward through the network and applying the chain rule to compute gradients.    

The goal of this step is to find the derivatives of the loss function with respect to every parameter of every layer (weights and biases), which will be used to update these parameters in the next step.

The approach used in this project was, after applying the chain rule, identifying there's a global gradient and a local gradient for every loss function gradient we are computing.

For example, in a network with 3 layers:
$$\frac{\partial L}{\partial w_3} = \frac{\partial L}{\partial a_3}  \frac{\partial a_3}{\partial z_3} \frac{\partial z_3}{\partial w_3}  $$
$$\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial a_3}  \frac{\partial a_3}{\partial z_3} \frac{\partial z_3}{\partial a_2}  \frac{\partial a_2}{\partial z_2}\frac{\partial z_2}{\partial w_2}  $$
$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial a_3}  \frac{\partial a_3}{\partial z_3} \frac{\partial z_3}{\partial a_2}  \frac{\partial a_2}{\partial z_2}\frac{\partial z_2}{\partial a_1}\frac{\partial a_1}{\partial z_1}\frac{\partial z_1}{\partial w_1}  $$
there is a global term that propagates from one layer to all other layers before this one, that's why the code computes every gradient using the global/local notation:
$$global_3 = \frac{\partial L}{\partial a_3}  \frac{\partial a_3}{\partial z_3} \ ,\  local_3 = \frac{\partial z_3}{\partial w_3}$$
$$global_2 = \frac{\partial z_3}{\partial a_2}  \frac{\partial a_2}{\partial z_2} \ ,\  local_2 = \frac{\partial z_2}{\partial w_2}$$
$$global_1 = \frac{\partial z_2}{\partial a_1}  \frac{\partial a_1}{\partial z_1} \ ,\  local_1 = \frac{\partial z_1}{\partial w_1}$$
the gradients with respect to the weights can therefore be calculated as follows:
$$\frac{\partial L}{\partial w_3} = global_3 \ local_3 \ $$
$$\frac{\partial L}{\partial w_2} = global_3 \ global_2 \ local_2$$
$$\frac{\partial L}{\partial w_1} = global_3 \ global_2 \ global_1 \ local_1$$
for the biases, the local gradient is 1, since the pre-activations $z$ are constant with respect to the biases. 
  
      
#### Gradient Descent 
Update the network parameters using gradient descent or its variants. This step involves adjusting the parameters in the direction that minimizes the loss. This project allows 2 methods of optimization:
1. **'sgd' : stochastic gradient descent**
$$\phi_{t+1} \leftarrow \phi_{t} - \alpha \frac{\partial L}{\partial \phi} $$
where $\phi$ is the parameter updated and $\alpha$ is the learning rate.

2. **'adam' : adaptative moment estimation** 
the adam method considers the momentum of the gradients normalized, so that it will move at fixed step sizes at every point.
$$m_{t+1} \leftarrow  \beta_1 m_t + (1 - \beta_1 )  \frac{\partial L[\phi_t]}{\partial \phi}$$
$$v_{t+1} \leftarrow \beta_2 v_t + (1 - \beta_2) {(\frac{\partial L[\phi_t]}{\partial \phi})}^2$$
$$\hat m_{t+1} \leftarrow  \frac{m_{t+1}}{1 - \beta_1^{t+1}}$$
$$\hat v_{t+1} \leftarrow  \frac{v_{t+1}}{1 - \beta_2^{t+1}}$$
$$\phi_{t+1} \leftarrow \phi_{t} - \alpha \frac{\hat m_{t+1} }{\sqrt{\hat v_{t+1}}+ \epsilon} \ , \ \epsilon = 10^{-7}$$
 
#### Training Loop 
Repeat steps for multiple ***epochs*** until convergence. Monitor the training loss to assess the model's performance during training.    
    
## Installation 
1. Clone the repository:    
```commandline 
git clone https://github.com/pedroroussos/neuralnetwork.git   
``` 
2. Navigate to the project directory:    
```commandline 
cd neuralnetwork   
``` 
3. Install dependencies:    
```commandline 
pip install numpy nptyping scikit-learn   
```    
 ## Usage 
 To use the neural network framework, follow these steps:    
1. Import the `Network` and `Layer` classes from the appropriate modules.    
2. Create a `Network` object by specifying the network architecture and hyperparameters.    
3. Add layers to the network using the `add_layer` method, specifying the number of units and activation function for each layer.    
4. Train the network using the `train` method, providing input and output data.    
5. Evaluate the trained model using the `predict` method or other evaluation metrics.    
    
Example usage:    
```python 
from network.Network import Network
    
 # Create a neural network object 
network = Network(input_size=64, 
                  loss_function='cce', 
                  init_method='xavier', 
                  batch_size=2, 
                  n_epochs=1000, 
                  learning_rate=0.01, 
                  optimizer='adam')    
 
 # Add layers to the network 
network.add_layer(units=128, activation='relu') 
network.add_layer(units=64, activation='sigmoid') 
network.add_layer(units=10, activation='softmax')  

 # Train the network 
network.train(x_train, y_train) 

 # Evaluate the trained model 
accuracy = network.compute_accuracy(x_test, y_test) 
print("Accuracy:", accuracy)   
```    
 ## Contributing 
 Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.    
    
## Contact 
For any questions or inquiries about this project, feel free to contact me at pedroroussos@gmail.com.