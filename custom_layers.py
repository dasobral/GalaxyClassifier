import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Layer
from keras import backend as K
import pickle

class Maxout(Layer):
    
    """
    Implements the maxout operation, where the output is the maximum of a 
    group of inputs. This layer is useful in neural networks to add non-linearity.
    """
    
    def __init__(self, group_size, **kwargs):
        
        """
        Initialize the Maxout layer.
        
        Parameters
        ----------
        group_size : int
            The number of inputs in each group. The output will be the maximum 
            of each group of inputs.
            
        Notes
        -----
        The __init__ function is special in Python classes. It gets called 
        when you create a new instance of the class. In this case, it is initializing 
        the group_size attribute and then calling the parent's __init__ function.
        """
        
        self.group_size = group_size
        super(Maxout, self).__init__(**kwargs)

    def build(self, input_shape):
        
        """
        Build the Maxout layer.
        
        Parameters
        ----------
        input_shape : tuple
            The shape of the inputs to this layer.
        
        Notes
        -----
        The purpose of this method is to validate if the last dimension of the input shape 
        is divisible by the group_size. If not, it raises a ValueError. This function is specific 
        to keras.Layer subclasses, which helps in creating the weight variables.
        """

        if input_shape[-1] % self.group_size != 0:
            raise ValueError("The last dimension of the inputs to `Maxout` "
                             "should be divisible by `group_size`. "
                             "Found `group_size` = {} and "
                             "last dimension of input shape = {}."
                             .format(self.group_size, input_shape[-1]))
        super(Maxout, self).build(input_shape)

    def call(self, inputs):
        
        """
        Call the Maxout layer.
        
        Parameters
        ----------
        inputs : Tensor
            The inputs to the layer.
        
        Returns
        -------
        outputs : Tensor
            The output of the maxout operation.
            
        Notes
        -----
        The call method defines the layer’s computation. For Maxout, this involves reshaping 
        the input tensor and applying the max operation. 
        """
        
        input_shape = K.shape(inputs)
        new_shape = K.concatenate([input_shape[:-1], [input_shape[-1] // self.group_size, self.group_size]])
        x = K.reshape(inputs, new_shape)
        outputs = K.max(x, axis=-1)
        return outputs

    def compute_output_shape(self, input_shape):
        
        """
        Compute the output shape of the Maxout layer.
        
        Parameters
        ----------
        input_shape : tuple
            The shape of the inputs to this layer.
        
        Returns
        -------
        shape : tuple 
            The shape of the output of this layer.
            
        Notes
        -----
        This function calculates the shape of the output of the layer. 
        This is required in Keras, to inform the model about the output size of the layer.
        """
        
        shape = list(input_shape)
        shape[-1] = shape[-1] // self.group_size
        return tuple(shape)
    
    def get_config(self):
        
        """
        Get the configuration of the Maxout layer.
        
        Returns
        -------
        config : dict
            A dictionary containing the configuration of the Maxout layer.
        
        Notes
        -----
        The get_config method is used when saving the model. It returns a 
        dictionary containing the configuration of the layer.
        """
        
        config = super().get_config()
        config.update({
            "group_size": self.group_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        
        """
        Create a Maxout layer from its configuration.
        
        Parameters
        ----------
        config : dict
            A dictionary containing the configuration of the Maxout layer.
            
        Returns
        -------
        maxout : Maxout
            A new Maxout layer instance.
            
        Notes
        -----
        This is a class method that is used to create a new instance of the 
        class using the configuration dictionary returned by get_config. 
        This is essential for Keras' model saving and loading functionality.
        """
        
        return cls(**config)
    

class MultiOutputDataGenerator(Sequence):
    
    """
    MultiOutputDataGenerator class extends the Sequence class in Keras to generate batches of tensor image 
    data with real-time data augmentation. The data will be looped over (in batches) indefinitely. 
    This class allows the handling of multiple output labels and uses the Keras ImageDataGenerator for the 
    actual data augmentation.

    Parameters
    ----------
    x_set : np.ndarray
        Input data. 
    y_set : list of np.ndarray or np.ndarray
        List of output data in case of multiple outputs or single output data in case of single output.
    batch_size : int
        Number of samples per gradient update.
    parameters : dict
        Parameters to be used for data augmentation by the internal ImageDataGenerator instance. 
        For more details on the parameters refer to the documentation of ImageDataGenerator.

    Notes
    -----
    In this class, we assume that `y_set` is a list of numpy arrays for multiple outputs or a single numpy 
    array for a single output. If you are providing a different format for `y_set`, you might need to adjust 
    the `__getitem__` method accordingly.
    """
    
    def __init__(self, x_set, y_set, batch_size, parameters, shuffle = True):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])
        self.aug_data = ImageDataGenerator(**parameters)
        self.shuffle = shuffle
        self.aug_data.fit(x_set)

    def __len__(self):
        
        """
        Get the number of batches per epoch.

        Returns
        -------
        int
            The number of batches per epoch.
        """
        
        return int(np.ceil(len(self.x) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        
        """
        Generates a batch of augmented data.

        This method is called for every mini-batch and it should return a complete batch each time. 
        It allows the generator to give the model a type of 'view' on the dataset, returning a batch of 
        inputs (and targets) at each call. In this implementation, the method also performs data augmentation 
        using the ImageDataGenerator instance.

        Parameters
        ----------
        idx : int
            Index of the mini-batch. It ranges between 0 and the total number of batches.

        Returns
        -------
        tuple
            A tuple of two elements. The first element is a numpy array of input data after augmentation. 
            If there are multiple outputs, the second element is a list of numpy arrays corresponding to each 
            output. If there's a single output, the second element is a numpy array for that output.

        Notes
        -----
        For multi-output models, the second element of the tuple is a list of numpy arrays. The order of the 
        numpy arrays in the list is the same as the order of the outputs in the model.
        
        In the case of single-output models, the second element of the tuple is a single numpy array 
        corresponding to the output of the model.

        """
        
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[indices]

        if isinstance(self.y, list):
            # Multi-output case
            batch_y = [output[indices] for output in self.y]
        else:
            # Single-output case
            batch_y = self.y[indices]

        augmented_batch_x = np.zeros_like(batch_x)
        for i in range(batch_x.shape[0]):
            augmented_batch_x[i] = self.aug_data.random_transform(batch_x[i])

        return augmented_batch_x, batch_y


    def on_epoch_end(self):
        
        """
        Shuffle the data at the end of every epoch.

        Notes
        -----
        The method is called at the end of every epoch and is used to shuffle the input and output data to 
        ensure the model doesn't learn any unintended sequential patterns from the data.
        """
        if not self.shuffle:
            self.indices = np.arange(self.x.shape[0])
        else:
            np.random.shuffle(self.indices)
            
    def generate_augmented_images(self, n_images):
        augmented_images = []
        num_batches = int(np.ceil(n_images / self.batch_size))

        for _ in range(num_batches):
            batch_x = self.x[:self.batch_size]
            augmented_batch_x = self.aug_data.flow(batch_x, batch_size=self.batch_size, shuffle=self.shuffle)
            augmented_images.extend(augmented_batch_x)
            self.x = self.x[self.batch_size:]

        augmented_images = np.concatenate(augmented_images, axis=0)
        augmented_images = augmented_images[:n_images]

        return augmented_images

    def save(self, filename):
        
        """
        Save the current state of the data generator to a file.

        Parameters
        ----------
        filename : str
            The name of the file where the state of the data generator will be saved.
        """
        
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        
        """
        Load the state of a data generator from a file.

        Parameters
        ----------
        filename : str
            The name of the file from which the state of the data generator will be loaded.

        Returns
        -------
        MultiOutputDataGenerator
            A new instance of MultiOutputDataGenerator with its state loaded from the file.
        """
        
        with open(filename, 'rb') as f:
            data_generator = cls(None, None, None)
            data_generator.__dict__.update(pickle.load(f))
        return data_generator

