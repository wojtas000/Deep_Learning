import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torch.optim.lr_scheduler import StepLR
import datasets
from weighted_random_search import wrs
from tqdm import tqdm
from cnn_models import Simple_CNN


class Net_wrapper:

    """ 
    Wrapper for neural network model. It combines the model itself (nn.Module) together with
    optimizer, learning rate scheduler, loss function and other
    training parameters (such as max_epochs, learning rate and batch size). 
    It allows to perform training of neural network simply by creating instance of Net_wrapper and 
    running `score` method. Compatible with grid search and random search classes.
    """
        
    def __init__(self, model=Simple_CNN, criterion=nn.CrossEntropyLoss(), optimizer=optim.Adam, weight_decay = 0,
                 max_epochs=5, batch_size=32, learning_rate=0.001, step_size=10, gamma=0.5, **kwargs):
        """
        Args:
        model: Neural network model (e. g. torch.nn.Module)
        criterion: loss function
        optimizer: optimizer used for minimizing the loss function
        weight_decay: strength of weight regularization
        max_epochs: number of epochs for training
        batch_size: size of batches feeding the neural network
        learning_rate: learning rate
        step_size: number of epochs, after which the learning rate drops down
        gamma: parameter multiplied by learning rate after 'step_size' number of epochs.
        **kwargs: other parameters of the model. Note: the key of parameter must correspond to exact name of 
                  attribute in model. For example, if model has attribute 'number_of_filters', the passed parameter should 
                  have the exact same name, 'number of filters'.
        """

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
        Method implementing forward-backward propagation loop and computing accuracy on training and validation sets.
        Args:
        train_dataset: PyTorch dataset of training data
        val_dataset: PyTorch dataset of validation data
        verbose: if 1, additional information is printed after each epoch such as training/validation loss/accuracy
        Returns:
        Training accuracy, training loss, validation accuracy and validation loss after full training. 
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

        
        # Forward-backward propagation loop
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

    """
    Class implementing classic random search over the space of parameters.
    """

    def __init__(self, net: Net_wrapper, param_grid, verbose=1):
      """
      Args:
      net: Net_wrapper instance
      param_grid: dictionary of parameters that we search for. 
      verbose: if 1, print additional information after each trial, 
      such as set of parameters that was checked and model accuracy on chosen set of parameters.
      """
      self.net = net
      self.param_grid = param_grid
      self.scores = []
      self.best_score = 0
      self.best_params = {}
      self.verbose = verbose

     
    @staticmethod
    def choose_random__params(parameters, seed=1):
      
      """
      Helper function used for choosing parameter set at random.
      Args:
      parameters: parameter dictionary.
      seed: seed of random state.
      """

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
      Fit the random search with train and validation dataset. 
      Search for optimal parameters for neural network declared during 
      initialization of RandomSearch instance.
      Args:
      train_dataset: PyTorch Dataset of training data
      cal_dataset: PyTprch Dataset of validation data
      n_trials: number of trials performed during random search.
      
      Returns:
      self
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
    Class used to perform grid search on set of parameters. 
    """

    def __init__(self, net: Net_wrapper, param_grid, step_by_step=False, verbose=1):
        """
        Args:
        net: Net_wrapper instance
        param_grid: dictionary of parameters
        step_by_step: if False normal grid search is performed, if True each parameter is evaluated step by step 
                      (not all the parameters together)
        verbose: if 1, additional information (parameter set and accuracy) prints with each iteration of grid search. 
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
        Args:
        train_dataset: PyTorch instance of train data
        val_dataset: PyTorch instance of validation data
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
    Class used to perform weighted random search on neural networks.
    Not completed - unable to use fANOVA package. 

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
        Args:
        net - Net_wrapper instance
        param_grid - dictionary of parameters we want to search
        verbose - if set to 1 additional information (parameter set and accuracy) prints with each iteration of grid search. 
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
    
# GridSearch and RandomSearch tests

if __name__=="__main__":
    
    train_dataset = datasets.cifar_train
    val_dataset = datasets.cifar_val

    subset_indices = list(range(500))
    subset_sampler = SubsetRandomSampler(subset_indices)

    subset_train_dataset = Subset(train_dataset, subset_indices)
    subset_val_dataset = Subset(val_dataset, subset_indices)

    test_hyper_params = {'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01], 'batch_size': [8, 16, 32, 64, 128]}
    my_net = Net_wrapper(model=Simple_CNN, max_epochs=5)
    
    gs = GridSearch(net=my_net, param_grid=test_hyper_params, step_by_step=True, verbose=1)
    gs = gs.fit(subset_train_dataset, subset_val_dataset)

    print(gs.best_score)
    print(gs.best_params)

    rs = RandomSearch(my_net, test_hyper_params, verbose=1)
    rs.fit(subset_train_dataset, subset_val_dataset, n_trials = 5)

    print(rs.best_score)
    print(rs.best_params)
