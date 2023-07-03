from cnn_models import Simple_CNN
import torch
import numpy as np

class Ensemble:
    """
    Ensemble of models.
    """

    def __init__(self,model_class=Simple_CNN, model_paths=None):
        
        """
        Args:
            model_class (class): Class of model to be loaded.
            model_paths (list): Paths to models.
        """

        self.models = []
        for model_path in model_paths:
            model = model_class()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            self.models.append(model)
    
    def predict_mean(self, test_dataset):
        """
        Predict the model using mean method.
        Args:
            test_Dataset (tf.data.Dataset): Test dataset.
        Returns:
            np.ndarray: Predicted labels.
        """
        predictions = []
        for model in self.models:
            predictions.append(model.predict(test_dataset).detach().numpy())
        predictions = np.array(predictions)
        return np.argmax(np.mean(predictions, axis=0), axis=1)
 
    
    def predict_max(self, test_dataset):
        """
        Predict the model using max method.
        Args:
            test_Dataset (tf.data.Dataset): Test dataset.
        Returns:
            np.ndarray: Predicted labels.
        """
        predictions = []
        for model in self.models:
            predictions.append(model.predict(test_dataset).detach().numpy())
        predictions = np.array(predictions)
        return np.argmax(np.max(predictions, axis=0), axis=1)
    

if __name__ == "__main__":
    model_paths = ["saved_models/model1.pt", "saved_models/model2.pt", "saved_models/model3.pt"]
    ensemble = Ensemble(model_paths=model_paths)
    test_dataset = torch.rand(100, 3, 32, 32) * 255
    print(ensemble.predict_mean(test_dataset))
    print(ensemble.predict_max(test_dataset))