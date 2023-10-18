import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from keras import backend as K
from keras.losses import Loss

class FocalLoss(Loss):
    
    """
    Implements the Focal Loss function, which is particularly useful for 
    a classification task when there is an extreme imbalance between the classes.

    Attributes
    ----------
    alpha : float, optional
        The scaling factor for the loss. Default is 0.25.
        
    gamma : float, optional
        The focusing parameter. Default is 2.0.
        
    alpha_threshold : float, optional
        The threshold for the alpha value. Default is 0.05.
        
    class_weight_dict : dict, optional
        A dictionary that maps each class to a custom weight. Default is None.

    """
    
    def __init__(self, alpha=0.2, gamma=1.5, alpha_threshold=0.05, class_weight_dict=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.alpha_threshold = alpha_threshold
        self.class_weight_dict = class_weight_dict

    def focal_loss(self, y_true, y_pred):
        
        """
        Implements the logic of the Focal Loss function.

        Parameters
        ----------
        y_true : Tensor
            The true labels.
        
        y_pred : Tensor
            The predicted labels.

        Returns
        -------
        focal_loss : Tensor
            The computed Focal Loss.

        Note
        -----
        This function is not intended to be used directly. Instead, it's used in the `call` method.
        """
        
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        if self.class_weight_dict is not None:
            alpha_weights = tf.gather(list(self.class_weight_dict.values()), tf.argmax(y_true, axis=1))
            alpha_weights = tf.reshape(alpha_weights, [-1, 1])  # reshaping to match y_pred and y_true shape
            focal_loss_positive = -alpha_weights * K.pow(1.0 - y_pred, self.gamma) * y_true * K.log(y_pred + epsilon)
            focal_loss_negative = -alpha_weights * K.pow(y_pred, self.gamma) * (1 - y_true) * K.log(1-y_pred + epsilon)
        else:
            focal_loss_positive = -self.alpha * K.pow(1.0 - y_pred, self.gamma) * y_true * K.log(y_pred + epsilon)
            focal_loss_negative = -self.alpha * K.pow(y_pred, self.gamma) * (1 - y_true) * K.log(1-y_pred + epsilon)
            
        focal_loss = focal_loss_positive + focal_loss_negative
            
        return K.mean(focal_loss, axis=-1)

    def call(self, y_true, y_pred):
        
        """
        Calls the Focal Loss function.

        Parameters
        ----------
        y_true : Tensor
            The true labels.
        
        y_pred : Tensor
            The predicted labels.

        Returns
        -------
        focal_loss : Tensor
            The computed Focal Loss.
        """
        
        loss = self.focal_loss(y_true, y_pred)

        return loss
    
    def get_config(self):
        
        """
        Gets the configuration of the FocalLoss class.

        Returns
        -------
        config : dict
            A dictionary containing the configuration of the FocalLoss class.
        """
        
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
            "alpha_threshold": self.alpha_threshold,
            "class_weight_dict": self.class_weight_dict
        })
        return config

    @classmethod
    def from_config(cls, config):
        
        """
        Creates a FocalLoss class from its configuration.

        Parameters
        ----------
        config : dict
            A dictionary containing the configuration of the FocalLoss class.

        Returns
        -------
        FocalLoss
            A new FocalLoss class.

        Note
        -----
        This is a class method that returns a new FocalLoss class given its configuration. 
        """
        
        return cls(**config)
    

class WeightedMeanSquaredErrorLoss(Loss):
    
    """
    Implements the Weighted Mean Squared Error (WMSE) loss function. This class allows to set custom 
    weights for each sample which is particularly useful when the samples are imbalanced.

    Attributes
    ----------
    class_weights : dict
        A dictionary that maps each output to a custom weight.
    """
    
    def __init__(self, class_weights, **kwargs):
        
        """
        Initializes the WeightedMeanSquaredErrorLoss class.
        
        Parameters
        ----------
        class_weights : dict
            A dictionary mapping each output to its weight.
        kwargs : dict
            Additional keyword arguments.
        """
        
        super().__init__(**kwargs)
        self.class_weights_tensor = tf.convert_to_tensor([weights for weights in class_weights.values()], dtype=tf.float32)

    def call(self, y_true_list, y_pred_list):
        """
        Computes the weighted mean squared error loss for each output.

        Parameters
        ----------
        y_true_list : list of Tensors
            The true labels for each output.
        y_pred_list : list of Tensors
            The predicted labels for each output.

        Returns
        -------
        loss_list : list of Tensors
            The computed weighted mean squared error loss for each output.
        """
        mse = tf.keras.losses.MeanSquaredError()
        def calculate_loss(args):
            i, y_true, y_pred = args[0], args[1], args[2]
            loss = mse(y_true, y_pred)
            weight_vector = tf.gather(self.class_weights_tensor, i)
            weighted_loss = loss * weight_vector
            return tf.reduce_mean(weighted_loss)

        loss_list = tf.map_fn(calculate_loss, (tf.range(tf.shape(self.class_weights_tensor)[0]), y_true_list, y_pred_list), fn_output_signature=tf.float32)
        return loss_list

    def get_config(self):
        
        """
        Gets the configuration of the WeightedMeanSquaredErrorLoss class.

        Returns
        -------
        config : dict
            A dictionary containing the configuration of the WeightedMeanSquaredErrorLoss class.
        """
        
        config = super().get_config()
        config.update({
            "class_weights": {str(i): weights.numpy().tolist() for i, weights in enumerate(self.class_weights_tensor)}
        })
        return config

    @classmethod
    def from_config(cls, config):
        
        """
        Creates a WeightedMeanSquaredErrorLoss class from its configuration.

        Parameters
        ----------
        config : dict
            A dictionary containing the configuration of the WeightedMeanSquaredErrorLoss class.

        Returns
        -------
        WeightedMeanSquaredErrorLoss
            A new WeightedMeanSquaredErrorLoss class instance.

        Note
        -----
        This is a class method that returns a new WeightedMeanSquaredErrorLoss class instance
        given its configuration. 
        """
        
        config["class_weights"] = {int(key): tf.convert_to_tensor(weights, dtype=tf.float32) 
                                   for key, weights in config["class_weights"].items()}
        return cls(**config)


class WeightedCategoricalCrossentropy(Loss):
    
    """
    Implements the Weighted Categorical Cross Entropy (WCCE) loss function. This class allows to set custom 
    weights for each class, which is particularly useful when dealing with imbalanced classes.

    Attributes
    ----------
    class_weights : dict
        A dictionary that maps each class to a custom weight.
    """
    
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.class_weights_tensor = tf.constant(list(class_weights.values()), dtype=tf.float32)

    def weighted_categorical_crossentropy(self, y_true, y_pred):
        
        """
        Computes the weighted categorical cross entropy loss.

        Parameters
        ----------
        y_true : Tensor
            The true labels.
        
        y_pred : Tensor
            The predicted labels.

        Returns
        -------
        weighted_loss : Tensor
            The computed weighted categorical cross entropy loss.

        Note
        -----
        This function is not intended to be used directly. Instead, it's used in the `call` method.
        """
        
        weights = tf.gather(self.class_weights_tensor, tf.argmax(y_true, axis=1))
        loss = K.categorical_crossentropy(y_true, y_pred)
        weighted_loss = loss * weights
        return tf.reduce_mean(weighted_loss)

    def call(self, y_true, y_pred):
        
        """
        Calls the weighted categorical cross entropy loss function.

        Parameters
        ----------
        y_true : Tensor
            The true labels.
        
        y_pred : Tensor
            The predicted labels.

        Returns
        -------
        weighted_loss : Tensor
            The computed weighted categorical cross entropy loss.
        """
        
        return self.weighted_categorical_crossentropy(y_true, y_pred)
    
    def get_config(self):
        
        """
        Gets the configuration of the WeightedCategoricalCrossentropy class.

        Returns
        -------
        config : dict
            A dictionary containing the configuration of the WeightedCategoricalCrossentropy class.
        """
        
        config = super().get_config()
        config.update({
            "class_weights": self.class_weights
        })
        return config

    @classmethod
    def from_config(cls, config):
        
        """
        Creates a WeightedCategoricalCrossentropy class from its configuration.

        Parameters
        ----------
        config : dict
            A dictionary containing the configuration of the WeightedCategoricalCrossentropy class.

        Returns
        -------
        WeightedCategoricalCrossentropy
            A new WeightedCategoricalCrossentropy class instance.

        Note
        -----
        This is a class method that returns a new WeightedCategoricalCrossentropy class instance
        given its configuration. 
        """
        
        return cls(**config)