import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import ParameterGrid
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset
from torch.optim.lr_scheduler import StepLR
import datasets
from weighted_random_search import wrs
from tqdm import tqdm


class ConvolutionalNeuralNetwork():
    
    def train_step(self, data, optimizer, criterion):
        x, y = data

        optimizer.zero_grad()

        logits = self(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        accuracy = (logits.argmax(dim=1) == y).float().mean()

        return {'loss': loss, 'accuracy': accuracy}

    
    def test_step(self, data, criterion):
        x, y = data

        logits = self(x)
        loss = criterion(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()

        return {'loss': loss, 'accuracy': accuracy}
    
    def Conv2d_output_size(self, w, k, s, p):
        '''
        w - width of input image
        k - kernel size
        s - stride
        p - padding
        '''
        return (w - k + 2 * p) / s + 1
    

# class CNN_3_class(nn.Module, ConvolutionalNeuralNetwork):
#     def __init__(self, num_classes = 10
#                     ,kernel_size1=3
#                     ,kernel_size2=3
#                     ,stride=1
#                     ,padding=1
#                     ,number_of_filters0=32
#                     ,number_of_filters1=32
#                     ,length_of_input0=32
#                     ,no_neurons = 128
#                     ,dr=nn.Dropout(p=0)
#                     ,activation_function=torch.relu):
#         super(CNN_3_class, self).__init__()
#         self.conv1 = nn.Conv2d(3, number_of_filters0, kernel_size1, stride, padding)
#         self.pool1 = nn.MaxPool2d(2)
#         length_of_input1 = self.Conv2d_output_size(length_of_input0, kernel_size1, stride, padding)/2
#         self.conv2 = nn.Conv2d(number_of_filters0, number_of_filters1, kernel_size2, stride, padding)
#         self.pool2 = nn.MaxPool2d(2)
#         length_of_input2 = self.Conv2d_output_size(length_of_input1, kernel_size2, stride, padding)/2
#         self.fc1 = nn.Linear(int(number_of_filters1*length_of_input2*length_of_input2), no_neurons)
#         self.dr = dr
#         self.fc2 = nn.Linear(no_neurons, num_classes)
#         # parameters
#         self.num_classes = num_classes
#         self.kernel_size1 = kernel_size1
#         self.kernel_size2 = kernel_size2
#         self.stride = stride
#         self.padding = padding
#         self.number_of_filters0 = number_of_filters0
#         self.number_of_filters1 = number_of_filters1
#         self.length_of_input0 = length_of_input0
#         self.no_neurons = no_neurons

#         self.activation_function = activation_function
        
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.activation_function(x)
#         x = self.pool1(x)
#         length_of_input1 = self.Conv2d_output_size(self.length_of_input0, self.kernel_size1, self.stride, self.padding)/2
#         x = self.conv2(x)
#         x = self.activation_function(x)
#         x = self.pool2(x)
#         length_of_input2 = self.Conv2d_output_size(length_of_input1, self.kernel_size2, self.stride, self.padding)/2
#         x = x.view(-1, int(self.number_of_filters1*length_of_input2*length_of_input2))
#         x = self.fc1(x)
#         x = self.activation_function(x)
#         x = self.dr(x)
#         x = self.fc2(x)
#         return x
    

class Net_wrapper:
    """ 
    Wrapper for neural network model. It combines the model itself (nn.Module) together with
    optimizer, loss function and training parameters (such as max_epochs, learning rate and batch size)
    """
        
    def __init__(self, model=CNN_3_class, criterion=nn.CrossEntropyLoss(), optimizer=optim.Adam, weight_decay = 0,
                 max_epochs=5, batch_size=32, learning_rate=0.001, step_size=10, gamma=0.5, **kwargs):
        if kwargs:
            self.model_params = kwargs
        else:
            self.model_params = {}
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scheduler_step_size=step_size
        self.scheduler_gamma=gamma
        self.weight_decay = weight_decay
    
    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def score(self, train_dataset, val_dataset, verbose=0):
        """
        Train model on train_dataset and calculate validation acurracy on val_dataset. 
        """
        if self.model_params:
            model = self.model(**self.model_params)
            
        else:
            model = self.model()
        if self.optimizer==optim.SGD:
            optimizer = self.optimizer(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        else:
            optimizer = self.optimizer(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = StepLR(optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        
        # TRAINING LOOP
        for epoch in range(self.max_epochs):
            
            if verbose == 1:
                print(f'Epoch {epoch+1}/{self.max_epochs}')
                t_loader = tqdm(train_loader)
                v_loader = tqdm(val_loader)
            else:
                t_loader = train_loader
                v_loader = val_loader
           
           # Train the model
            avg_train_accuracy = 0
            avg_train_loss = 0
            
            for x, y in t_loader:
                optimizer.zero_grad()

                output = model(x)
                train_loss = self.criterion(output, y)
                train_loss.backward()
                optimizer.step()

                train_accuracy = (output.argmax(dim=1) == y).float().mean()

                avg_train_accuracy += train_accuracy.item()
                avg_train_loss += train_loss.item()
            
            # Calculate average training loss and accuracy for the epoch
            avg_train_accuracy = avg_train_accuracy / len(train_loader)
            avg_train_loss = avg_train_loss / len(train_loader)
            
            if verbose == 1:
                print(f'train_acc:{avg_train_accuracy}, train_loss: {avg_train_loss}')

            # Test the model
            avg_val_loss = 0
            avg_val_accuracy = 0
            
            with torch.no_grad():
                for x, y in v_loader:
                    output = model(x)
                    val_loss = self.criterion(output, y)
                    val_accuracy = (output.argmax(dim=1) == y).float().mean()
                    avg_val_loss += val_loss.item()
                    avg_val_accuracy += val_accuracy.item()

            # Calculate average test loss and accuracy for the epoch
            avg_val_accuracy = avg_val_accuracy / len(val_loader)
            avg_val_loss = avg_val_loss / len(val_loader)

            if verbose == 1:
                print(f'val_acc:{avg_val_accuracy}, val_loss: {avg_val_loss}')

            scheduler.step()
        
        self.model = model
        
        return avg_train_accuracy, avg_train_loss, avg_val_accuracy, avg_val_loss


class RandomSearch:
    
    def __init__(self, net: Net_wrapper, param_grid, verbose=1):
      """

      """
      self.net = net
      self.param_grid = param_grid
      self.scores = []
      self.best_score = 0
      self.best_params = {}
      self.verbose = verbose

     
    @staticmethod
    def choose_random__params(parameters, seed=1):
      random_params = {}
      rnd = np.random.RandomState(seed)
      
      for param in parameters:

        if isinstance(parameters[param][0], float):
          random_params[param] = rnd.uniform(low=parameters[param][0], high=parameters[param][1])
        elif isinstance(parameters[param][0], int):
          random_params[param] = rnd.randint(low=parameters[param][0], high=parameters[param][1])
        else:
          random_params[param] = parameters[param][rnd.randint(0, len(parameters[param]))]
      
      return random_params

      
    def fit(self, train_dataset, val_dataset, n_trials = 10):
      """
      Fit the grid search with train and validation dataset. 
      Search for optimal parameters for neural network declared during 
      initialization of GridSearch instance.
      """
      
      for trial in range(n_trials):
          random_params = RandomSearch.choose_random__params(parameters=self.param_grid, seed=trial)
          for hyp_name, hyp_val in random_params.items():
              if hasattr(self.net, hyp_name):
                  setattr(self.net, hyp_name, hyp_val)
              else:
                  self.net.model_params[hyp_name] = hyp_val

          _, _, val_accuracy, _ = self.net.score(train_dataset, val_dataset)
          self.scores.append(val_accuracy)
          if val_accuracy > self.best_score:
              self.best_score = val_accuracy
              self.best_params = random_params
          if self.verbose == 1:
              print('Parameter set:', random_params)
              print(f'val_accuracy: {val_accuracy:.4f}')

      return self


class GridSearch:
    """
    Class used to perform grid search on neural networks 

    Attributes:
    self.net - Net_wrapper instance
    self.param_grid - dictionary of parameters we want to search
    self.scores - list for scores of each set of parameters
    self.best_score - best score out of all parameters
    self.best_params - best set of parameters
    self.verbose - if set to 1 additional information (parameter set and accuracy) prints with each iteration of grid search. 
    """
    def __init__(self, net: Net_wrapper, param_grid, step_by_step=False, verbose=1):
        """

        """
        self.net = net
        self.param_grid = param_grid
        self.scores = []
        self.best_score = 0
        self.best_params = {}
        self.step_by_step = step_by_step
        self.verbose = verbose


    def fit(self, train_dataset, val_dataset):
        """
        Fit the grid search with train and validation dataset. 
        Search for optimal parameters for neural network declared during 
        initialization of GridSearch instance.
        """
        if self.step_by_step==False:
            for params in ParameterGrid(self.param_grid):
                for hyp_name, hyp_val in params.items():
                    if hasattr(self.net, hyp_name):
                        setattr(self.net, hyp_name, hyp_val)
                    else:
                        self.net.model_params[hyp_name] = hyp_val

                _, _, val_accuracy, _ = self.net.score(train_dataset, val_dataset)
                self.scores.append(val_accuracy)
                if val_accuracy > self.best_score:
                    self.best_score = val_accuracy
                    self.best_params = params
                if self.verbose == 1:
                    print('Parameter set:', params)
                    print(f'val_accuracy: {val_accuracy:.4f}')
        else:
            
            for hyp_name, hyp_vals in self.param_grid.items():
                score = 0
                for hyp_val in hyp_vals:
                    if hasattr(self.net, hyp_name):
                        setattr(self.net, hyp_name, hyp_val)
                    else:
                        self.net.model_params[hyp_name] = hyp_val 

                    _, _, val_accuracy, _ = self.net.score(train_dataset, val_dataset)
                    self.scores.append(val_accuracy)
                    
                    if val_accuracy > score:
                        if score > self.best_score:
                            self.best_score = score
                        score = val_accuracy
                        self.best_params[hyp_name] = hyp_val
                    
                    if self.verbose == 1:
                        print(f'Current parameter: {hyp_name}:', hyp_val, f'  val_accuracy: {val_accuracy:.4f}')
                        print(f'Best parameters till now:{self.best_params}')
                
                if hasattr(self.net, hyp_name):
                    setattr(self.net, hyp_name, self.best_params[hyp_name])
                    print(getattr(self.net, hyp_name))
                else:
                    self.net.model_params[hyp_name] = self.best_params[hyp_name]

        return self
    

class WeightedRandomSearch():
    """
    Class used to perform grid search on neural networks 

    Attributes:
    self.net - Net_wrapper instance
    self.param_grid - dictionary of parameters we want to search
    self.scores - list for scores of each set of parameters
    self.best_score - best score out of all parameters
    self.best_params - best set of parameters
    self.verbose - if set to 1 additional information (parameter set and accuracy) prints with each iteration of grid search. 
    """
    def __init__(self, net, param_grid, verbose=1):
        """

        """
        self.net = net
        self.param_grid = param_grid
        self.scores = []
        self.best_score = 0
        self.best_params = None
        self.verbose = verbose

    def fit(self, train_dataset, val_dataset, N, N_0):
        """
        Fit the grid search with train and validation dataset. 
        Search for optimal parameters for neural network declared during 
        initialization of GridSearch instance.
        """
        def goal_function(params):
            
            for hyp_name, hyp_val in params.items():
                if hasattr(self.net, hyp_name):
                    setattr(self.net, hyp_name, hyp_val)
                else:
                    self.net.model_params[hyp_name] = hyp_val
            
            return self.net.score(train_dataset, val_dataset)[1]
            
        self.best_params, self.best_score = wrs(F=goal_function, N=N, N_0=N_0, param_grid=self.param_grid )
        
        return self
    

if __name__=="__main__":
    
    train_dataset = datasets.cifar_train
    val_dataset = datasets.cifar_val

    subset_indices = list(range(500))
    subset_sampler = SubsetRandomSampler(subset_indices)

    subset_train_dataset = Subset(train_dataset, subset_indices)
    subset_val_dataset = Subset(val_dataset, subset_indices)

    test_hyper_params = {'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01], 'batch_size': [8, 16, 32, 64, 128]}
    my_net = Net_wrapper(model=CNN_3_class, max_epochs=5)
    
    gs = GridSearch(net=my_net, param_grid=test_hyper_params, step_by_step=True, verbose=1)
    gs = gs.fit(subset_train_dataset, subset_val_dataset)

    print(gs.best_score)
    print(gs.best_params)

    rs = RandomSearch(my_net, test_hyper_params, verbose=1)
    rs.fit(subset_train_dataset, subset_val_dataset, n_trials = 5)

    print(rs.best_score)
    print(rs.best_params)
