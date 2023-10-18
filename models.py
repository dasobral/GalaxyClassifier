import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


from tensorflow import keras, expand_dims
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, GlobalMaxPooling2D, Dense, Dropout, Lambda, Activation, BatchNormalization, Multiply, Concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras import backend as K
from custom_layers import Maxout, MultiOutputDataGenerator
from custom_losses import FocalLoss, WeightedMeanSquaredErrorLoss, WeightedCategoricalCrossentropy
import numpy as np

class GalaxyZooClassifier(object):
    
    """
    Custom classifier for the Galaxy Zoo challenge.

    This class builds a custom model architecture, allowing for base, hierarchical, and branch models.
    Different parameters allow for flexible customization of the architecture, such as number and type 
    of convolutional layers, dense layers, batch normalization, pooling method, and dropout.

    Attributes
    ----------
    in_shape : tuple
        The shape of the input data (e.g., for image data, (height, width, channels)).
    n_classes : int
        The number of classes for classification.
    conv_layers : list of tuples
        The parameters for each convolutional layer. Each tuple should specify (number of filters, kernel size).
    dense_units : list of int
        The number of dense units in each dense layer.
    batch_normalization : bool, optional
        Whether to use batch normalization in the model.
    activation : str, optional
        The activation function to use in the model.
    pool_size : tuple, optional
        The pool size for MaxPooling layers.
    flattening : str, optional
        The method to flatten the convolutional layers before passing to dense layers.
    dropout : float, optional
        The dropout rate for Dropout layers in the model.
    monitor : str, optional
        The metric to monitor for early stopping.
    class_weights : bool, optional
        Whether to compute class weights.
    out_activation : str, optional
        The activation function for the output layer.
    max_out : bool, optional
        Whether to use Maxout activation.
    model_type : str
        The type of the model to build, one of 'base', 'hierarchy', 'branch'.
    early_stopping : EarlyStopping object, optional
        Keras EarlyStopping callback to use during training.
    data_augmentation : dict, optional
        Parameters for data augmentation.
    model : keras.Model
        The model to be compiled and fit.

    Methods
    -------
    build_convolutional_blocks(input_layer)
        Builds the convolutional blocks of the model.
    build_dense_blocks(input_layer)
        Builds the dense blocks of the model.
    build_architecture(input_layer)
        Builds the architecture of the model by adding convolutional and dense blocks.
    build_model()
        Builds a base model.
    build_hierarchy_model(n_classes)
        Builds a hierarchical model.
    build_branch_model(n_classes)
        Builds a branch model.
    compile_model(learning_rate, loss_function, metrics=['accuracy'])
        Compiles the model.
    fit_model(X_train, y_train, X_val, y_val, learning_rate, loss_function = 'categorical_crossentropy', metrics=['accuracy'], batch_size=100, epochs=25, threshold = 0.1, take_weights_log=True)
        Fits the model to the training data.
    save_model(models_path, name, diagram = False)
        Saves the model and its diagram.
    """
    
    def __init__(self, model_type, input_shape, n_classes, conv_layers, dense_units, batch_normalization = False ,activation='relu',  
                 pool_size=(2,2), flattening = 'Flatten', class_weights = False, out_activation = 'softmax', 
                 dropout_rate=None, max_out=False, early_stop_patience=None, monitor='val_loss', data_augmentation=None):
        
        """
        Initializes a GalaxyZooClassifier object.

        Parameters
        ----------
        model_type: str
            The type of model to be built. Should be either 'base', 'hierarchy', or 'branch'.
        input_shape: tuple
            The shape of the input data.
        n_classes: int
            The number of classes for classification.
        conv_layers: list of tuples
            The configuration of the convolutional layers, with each tuple containing (number of filters, kernel size).
        dense_units: list of int
            The number of units for each dense layer.
        batch_normalization: bool, optional (default=False)
            Whether to use batch normalization.
        activation: str, optional (default='relu')
            The activation function to use in the layers.
        pool_size: tuple, optional (default=(2, 2))
            The pool size for the max pooling layers.
        flattening: str, optional (default='Flatten')
            The type of flattening to be used before the dense layers.
        class_weights: bool, optional (default=False)
            Whether to use class weights during training.
        out_activation: str, optional (default='softmax')
            The activation function to use in the output layer.
        dropout_rate: float, optional
            The dropout rate to use after each layer.
        max_out: bool, optional (default=False)
            Whether to use Maxout in the dense layers.
        early_stop_patience: int, optional
            The number of epochs with no improvement after which training will be stopped. If None, early stopping is not used.
        monitor: str, optional (default='val_loss')
            Quantity to be monitored for early stopping.
        data_augmentation: dict, optional
            The parameters for data augmentation. If None, data augmentation is not used.

        Raises
        ------
        ValueError:
            If `model_type` is not 'base', 'hierarchy', or 'branch'.
        """
        
        self.in_shape = input_shape
        self.n_classes = n_classes
        self.conv_layers = conv_layers
        self.dense_units = dense_units
        self.batch_normalization = batch_normalization
        self.activation = activation
        self.pool_size = pool_size
        self.flattening = flattening
        self.dropout = dropout_rate
        self.monitor = monitor
        self.class_weights = class_weights
        self.out_activation = out_activation
        self.max_out = max_out
        self.model_type = model_type
        
            
        if early_stop_patience:
            self.early_stopping=EarlyStopping(monitor=self.monitor, patience=early_stop_patience, restore_best_weights=True, verbose=1)
        else:
            self.early_stopping = None
        
        # Save the data augmentation parameters, if any
        self.data_augmentation = data_augmentation
        
        if self.model_type == 'base':    
            self.model = self.build_model()
        elif self.model_type == 'hierarchy':
            self.model = self.build_hierarchy_model(self.n_classes)
        elif self.model_type == 'branch':
            self.model = self.build_branch_model(self.n_classes)
        else:
            raise ValueError("Invalid model_type. Expected 'base', 'hierarchy', or 'branch'.")

        
    def build_convolutional_blocks(self, input_layer):
        
        """
        Builds a sequence of convolutional layers based on the `conv_layers` attribute.

        Parameters
        ----------
        input_layer: tensorflow.python.keras.engine.input_layer.InputLayer
            The input layer of the model.

        Returns
        -------
        x: tensorflow.python.keras.engine.base_layer
            The output of the last layer in the block.

        Notes
        -----
        This method builds a sequence of convolutional layers, each followed by an activation function, a max pooling layer, 
        and possibly a batch normalization layer, based on the `conv_layers` attribute. The number of convolutional layers,
        their filter sizes, and their kernel sizes are determined by `conv_layers`. After the last convolutional layer, 
        the output is flattened if `flattening` is 'Flatten', or global max pooling is applied if `flattening` is not 'Flatten'.
        """
        
        n_conv_layers = len(self.conv_layers)
        
        # Add Convolutional layers with MaxPooling
        
        x = input_layer
        
        if self.flattening == 'Flatten':
            for i in range(n_conv_layers):
                x = Conv2D(filters=self.conv_layers[i][0],
                       kernel_size=self.conv_layers[i][1])(x)
                
                if self.batch_normalization:
                    x = BatchNormalization()(x)
                    
                x = Activation(self.activation)(x)
                x = MaxPooling2D(pool_size=self.pool_size)(x)
            
            x = Flatten()(x)
        else:
            for i in range(n_conv_layers-1):
                x = Conv2D(filters=self.conv_layers[i][0],
                       kernel_size=self.conv_layers[i][1])(x)
                
                if self.batch_normalization:
                    x = BatchNormalization()(x)
                    
                x = Activation(self.activation)(x)
                x = MaxPooling2D(pool_size=self.pool_size)(x)
            
            x = Conv2D(filters=self.conv_layers[-1][0],
                       kernel_size=self.conv_layers[-1][1])(x)
            if self.batch_normalization:
                    x = BatchNormalization()(x)
            x = Activation(self.activation)(x)
            
            x = GlobalMaxPooling2D()(x)
            
        return x
        
    def build_dense_blocks(self, input_layer):
        
        """
        Builds a sequence of fully connected (dense) layers based on the `dense_units` attribute.

        Parameters
        ----------
        input_layer: tensorflow.python.keras.engine.base_layer
            The input layer of the model.

        Returns
        -------
        x: tensorflow.python.keras.engine.base_layer
            The output of the last layer in the block.

        Notes
        -----
        This method builds a sequence of dense layers, each followed by an activation function, and optionally a dropout layer,
        based on the `dense_units` attribute. The number of dense layers and their units are determined by `dense_units`. 
        If `max_out` is True, a maxout layer is applied after each dense layer. If `dropout` is not None, a dropout layer is 
        applied after each dense layer with `dropout` rate.
        """
        
        #Add fully connected layers
        
        n_dense_units = len(self.dense_units)
        x = input_layer
        
        for i in range(n_dense_units):
            
            if self.max_out:
                x = Dense(2*self.dense_units[i], activation=self.activation)(x)
                x = Maxout(group_size=2)(x)
            else:
                x = Dense(self.dense_units[i], activation=self.activation)(x)
                
            if self.dropout is not None:
                x = Dropout(self.dropout)(x)
                
        return x
    
    def build_architecture(self, input_layer):
        
        """
        Builds the architecture of the model by combining convolutional and dense blocks.

        Parameters
        ----------
        input_layer: tensorflow.python.keras.engine.base_layer
            The input layer of the model.

        Returns
        -------
        DenseBlocks: tensorflow.python.keras.engine.base_layer
            The output of the last layer in the dense block.

        Notes
        -----
        This method calls `build_convolutional_blocks` and `build_dense_blocks` methods to build the architecture 
        of the model. The output of the convolutional blocks is passed as input to the dense blocks. 
        """
        
        ConvBlocks = self.build_convolutional_blocks(input_layer)
        DenseBlocks = self.build_dense_blocks(input_layer=ConvBlocks)
        
        return DenseBlocks 
        
    def build_model(self):
        
        """
        Builds the base model with the defined architecture.

        Returns
        -------
        model: keras.Model
            The base model with the defined architecture.

        Notes
        -----
        This method first creates an input layer with the shape defined in self.in_shape.
        It then calls the `build_architecture` method to build the architecture of the model.
        The output layer is a dense layer with a number of units equal to the number of classes,
        and activation function defined in self.out_activation.
        """

        
        input_layer = Input(shape=self.in_shape)
        
        architecture = self.build_architecture(input_layer=input_layer)
        
        # Add output layer with name
        output_layer = Dense(self.n_classes, activation=self.out_activation, name='Class1')(architecture)
        
        model = keras.Model(inputs=input_layer, outputs=output_layer)
        
        return model
    
    def build_hierarchy_model(self, n_classes, embbedings = False):
        
        """
        Builds a model with a hierarchical output architecture.

        Parameters
        ----------
        n_classes: tuple
            A tuple specifying the number of classes for each output layer.

        Returns
        -------
        model: keras.Model
            The model with the defined hierarchical architecture.

        Notes
        -----
        This method first creates an input layer with the shape defined in self.in_shape.
        It then calls the `build_architecture` method to build the architecture of the model.
        The output layers are dense layers with number of units equal to the number of classes
        for each layer, and activation function defined in self.out_activation. These layers
        are then processed to enforce the class hierarchy. The final model has multiple outputs,
        one for each class hierarchy.
        """
        
        out1_classes, out2_classes, out3_classes = n_classes
        
        input_layer = Input(shape=self.in_shape)
        
        architecture = self.build_architecture(input_layer=input_layer)
        
        # Add output layers for each class hierarchy, with names
        
        if embbedings:
            # Embedding layers for Class2 and Class7
            embedding_dim = 64  # adjust the dimensionality based on your dataset
            embedding2 = Dense(embedding_dim, activation='sigmoid', name='embedding2')(expand_dims(out1[:, 1], axis=-1))
            embedding7 = Dense(embedding_dim, activation='sigmoid', name='embedding7')(expand_dims(out1[:, 0], axis=-1))

            # Concatenate embeddings with previous layers' features
            combined_features2 = Concatenate()([architecture, embedding2])
            combined_features7 = Concatenate()([architecture, embedding7])
            
            out2_ = Dense(out2_classes, activation=self.out_activation, name='Class2_in')(combined_features2)
            out3_ = Dense(out3_classes, activation=self.out_activation, name='Class3_in')(combined_features7)
        else:
            out1 = Dense(out1_classes, activation=self.out_activation, name='Class1')(architecture)
            out2_ = Dense(out2_classes, activation=self.out_activation, name='Class2_in')(architecture)
            out3_ = Dense(out3_classes, activation=self.out_activation, name='Class3_in')(architecture)

        # Post-process the predictions to enforce the class hierarchy
        class1_2_mask = Lambda(lambda x: x[:, 1], name='class1_2_mask')(out1)
        out2 = Multiply(name='Class2')([out2_, expand_dims(class1_2_mask, -1)])

        class1_1_mask = Lambda(lambda x: x[:, 0], name='class1_1_mask')(out1)
        out3 = Multiply(name='Class3')([out3_, expand_dims(class1_1_mask, -1)])
        
        output_layer = [out1, out2, out3]
        
        model = keras.Model(inputs=input_layer, outputs=output_layer) 
        
        return model
    
    def build_branch_model(self, n_classes):
        
        """
        Builds a model with a branching output architecture.

        Parameters
        ----------
        n_classes: tuple
            A tuple specifying the number of classes for each output layer.

        Returns
        -------
        model: keras.Model
            The model with the defined branching architecture.

        Notes
        -----
        This method first creates an input layer with the shape defined in self.in_shape.
        It then builds the first level of the model using the `build_architecture` method
        and adds an output layer. For the second level, it builds two branches each with
        its own architecture and output layer. The outputs of the first level and each 
        branch of the second level are concatenated and then processed with dense blocks.
        The outputs of the dense blocks are then processed to enforce the class hierarchy.
        The final model has multiple outputs, one for each branch.
        """
        
        out1_classes, out2_classes, out3_classes = n_classes
        
        input_layer = Input(shape=self.in_shape)
    
        # First level. Classifies Class1.1, Class1.2, Class1.3
        Level1 = self.build_architecture(input_layer=input_layer)
        out1 = Dense(out1_classes, activation='softmax', name='Class1')(Level1)
    
        # Second level
        
        # Branch that classifies Class2.1, Class2.2
        Level2_1 = self.build_convolutional_blocks(input_layer=input_layer)
        x1 = Concatenate()([out1, Level2_1])
        x1 = self.build_dense_blocks(x1)
        out2_ = Dense(out2_classes, activation=self.out_activation, name='Class2_in')(x1)
        
        # Branch that classfies Class7.1, Class7.2, Class7.3
        Level2_2 = self.build_convolutional_blocks(input_layer=input_layer)
        x2 = Concatenate()([out1, Level2_2])
        x2 = self.build_dense_blocks(x2)
        out3_ = Dense(out3_classes, activation=self.out_activation, name='Class3_in')(x2)
    
        # Apply masks to enforce hierarchy
        
        class1_2_mask = Lambda(lambda x: x[:, 1], name='class1_2_mask')(out1)
        out2 = Multiply(name='Class2')([out2_, expand_dims(class1_2_mask, -1)])

        class1_1_mask = Lambda(lambda x: x[:, 0], name='class1_1_mask')(out1)
        out3 = Multiply(name='Class3')([out3_, expand_dims(class1_1_mask, -1)])
    
        final_output = [out1, out2, out3]
    
        # Define the model
        model = keras.Model(inputs=input_layer, outputs=final_output)
    
        return model
        
    def compile_model(self, learning_rate, loss_function, metrics=['accuracy']):
        
        """
        Compiles the model with the specified parameters.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for the Adam optimizer.

        loss_function: str or dict
            The loss function to use. If it's a string, it is used for all outputs.
            If it's a dictionary, it should map output names to loss functions.

        metrics: list, optional
            The list of metrics to compute during training and testing. 
            Default is ['accuracy'].

        Raises
        ------
        Exception
            If the model has not been built before this method is called.

        Notes
        -----
        The method first checks if the model has been built. If not, it raises an exception.
        It then creates an Adam optimizer with the specified learning rate and compiles the model
        with this optimizer, the specified loss function, and the metrics. If there are multiple
        outputs and the loss function is a string, it uses this loss function for all outputs.
        If the loss function is a dictionary, it should map output names to loss functions.
        """
        
        if not hasattr(self, 'model'):
            raise Exception('You must build the model before compiling it.')

        optimizer = Adam(learning_rate=learning_rate)

        # If there are multiple outputs (class hierarchies), use a separate loss function for each
        if isinstance(loss_function, str):
            loss_function = {name: loss_function for name in self.model.output_names}

        self.model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
            
    def fit_model(self, X_train, y_train, X_val, y_val, learning_rate,
                  loss_function = 'categorical_crossentropy', metrics=['accuracy'], batch_size=100, epochs=25, threshold = 0.1, take_weights_log=True):
        
        """
        Fits the model to the training data.

        Parameters
        ----------
        X_train: array
            The training inputs.
        
        y_train: array
            The training targets.
            
        X_val: array
            The validation inputs.
        
        y_val: array
            The validation targets.

        learning_rate: float
            The learning rate to use for the Adam optimizer.

        loss_function: str or dict
            The loss function to use. If it's a string, it is used for all outputs.
            If it's a dictionary, it should map output names to loss functions.

        metrics: list, optional
            The list of metrics to compute during training and testing. 
            Default is ['accuracy'].
            
        batch_size: int, optional
            The number of samples per batch. Default is 100.
            
        epochs: int, optional
            The number of epochs to train the model. Default is 25.
        
        threshold: float, optional
            The threshold for the class weights. Default is 0.1.
            
        take_weights_log: bool, optional
            Whether to take the log of the class weights. Default is True.

        Raises
        ------
        Exception
            If the model has not been built and compiled before this method is called.

        Notes
        -----
        The method first checks if the model has been built and compiled. If not, it raises an exception.
        It then compiles the model with the specified parameters and fits it to the training data. It also computes 
        class weights if specified, and uses a separate loss function for each output if there are multiple outputs. 
        The training process can also include early stopping and data augmentation if specified.
        """

        
        if not hasattr(self, 'model'):
            raise Exception('You must build and compile the model before fitting it.')
        
        for j in range(len(metrics)):
            if metrics[j] == 'F1_score':
                metrics[j] = F1_score
                
        # If there are multiple outputs (class hierarchies), use a separate loss function for each
        if isinstance(loss_function, str):
            loss_function = {name: loss_function for name in self.model.output_names}

        if self.class_weights:
            print()
            print('--------------------------------- COMPUTING CLASS WEIGHTS ---------------------------------')
            print()
            if isinstance(y_train, list):
                class_weights = {}
                for i, y in enumerate(y_train):
                    class_weight_dict, _ = calculate_class_weights(y, threshold=threshold, take_weights_log=take_weights_log)
                    class_weights[self.model.output_names[i]] = {int(k): float(v) for k, v in class_weight_dict.items()}
            else:
                class_weight_dict, _ = calculate_class_weights(y_train, threshold=threshold, take_weights_log=take_weights_log)
                class_weights = {self.model.output_names[0]: {int(k): float(v) for k, v in class_weight_dict.items()}}
            for key, value in class_weights.items():
                print(key, ":", value)
            print()
            print('--------------------------------- WEIGHTING LOSS FUNCTION ---------------------------------')
            print()
        else:
            class_weights = {self.model.output_names[k] : None for k in range(len(self.model.output_names))}

        new_loss_function = loss_function.copy()
        for key in self.model.output_names:
            lf = loss_function[key]
            if lf == 'focal_loss':
                self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0, class_weight_dict=class_weights[key])
                new_loss_function[key] = self.focal_loss
            elif lf == 'weighted_mse':
                self.weighted_mse = WeightedMeanSquaredErrorLoss(class_weights=class_weights[key])
                new_loss_function[key] = self.weighted_mse
            elif lf == 'weighted_cce':
                self.weighted_categorical_crossentropy = WeightedCategoricalCrossentropy(class_weights=class_weights[key])
                new_loss_function[key] = self.weighted_categorical_crossentropy
                
        print()
        print('--------------------------------- TRAINING MODEL ---------------------------------')
        print()    

        self.compile_model(learning_rate=learning_rate, loss_function=new_loss_function, metrics=metrics)
            
        callbacks = []
        if self.early_stopping:
            callbacks.append(self.early_stopping)
            
        if self.data_augmentation:
            # Instantiate the data generator with the training data and batch size
            train_generator = MultiOutputDataGenerator(x_set= X_train, y_set =  y_train, 
                                                       batch_size=batch_size, 
                                                       parameters = self.data_augmentation, shuffle=True)
            steps_per_epoch = len(train_generator)
            history = self.model.fit(train_generator,
                                epochs=epochs,
                                validation_data = (X_val, y_val),
                                steps_per_epoch = steps_per_epoch,
                                callbacks = callbacks, verbose=1)
        else:
            history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                                validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
            
        
        return history

    def save_model(self, models_path, name, diagram = False):
        
        """
        Saves the model and optionally its architecture diagram.

        Parameters
        ----------
        models_path: str
            The path to the directory where the model should be saved.

        name: str
            The name of the model. The model will be saved as a .h5 file with this name.

        diagram: bool, optional
            Whether to save a diagram of the model's architecture. Default is False.
            If True, the diagram is saved as a .pdf file with the same name as the model.

        Notes
        -----
        The method first saves the model as a .h5 file in the specified directory.
        If the diagram flag is set to True, it also saves a diagram of the model's architecture as a .pdf file in the same directory.
        """
        
        model_path = os.path.join(models_path, name+'.h5')
        self.model.save(model_path)
        
        print()
        print(f'Model saved in {model_path}')
        
        if diagram:
            diagram_path = os.path.join(models_path, name+'.pdf')
            plot_model(model=self.model, to_file=diagram_path, show_shapes=True)
            print()
            print(f'Diagram saved in {diagram_path}')
        
def F1_score(y_true, y_pred):
    
    """
    Compute the F1 Score between the true and predicted labels.

    The F1 score is the harmonic mean of precision and recall. Compared to the regular mean, 
    the harmonic mean gives much more weight to low values. The best value is 1 and the worst is 0.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated targets as returned by a classifier.

    Returns
    -------
    f1_sc : float
        The F1 Score between 0 and 1.

    Notes
    -----
    The F1 score is especially useful for balanced datasets, as it takes both false positives 
    and false negatives into account.
    """
    
    true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    pred_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
    precision = true_positives / (pred_positives + K.epsilon())
    recall = true_positives / (all_positives + K.epsilon())
    
    f1_sc = 2*(precision * recall)/(precision+recall+K.epsilon())
    
    return f1_sc
F1_score.__name__ = 'F1_score'
    
def calculate_class_weights(y_train, threshold, take_weights_log=True):
    
    """
    Calculate class weights for imbalanced datasets. The function only considers classes that have a representation
    above a certain threshold.

    Parameters
    ----------
    y_train : numpy.ndarray
        The one-hot encoded target variables for the training set. Shape should be (n_samples, n_classes).
    threshold : float
        The minimum representation threshold for a class to be considered valid. Should be between 0 and 1.

    Returns
    -------
    class_weight_dict : dict
        A dictionary mapping from class index to weight. Only includes classes above the representation threshold.
    valid_classes : numpy.ndarray
        An array of the classes that are above the representation threshold.

    Notes
    -----
    If a class is below the representation threshold, it is not included in the output dictionary. 
    """
    
    # Convert one-hot encoded y_train to label encoded
    y_train_labels = np.argmax(y_train, axis=1)

    # Calculate the total number of samples
    total_samples = len(y_train_labels)

    # Calculate the number of samples in each class
    class_samples = np.bincount(y_train_labels)

    # Identify valid classes (those with representation above the threshold)
    valid_classes = np.where(class_samples / total_samples > threshold)[0]
    
    # Calculate class weights only for valid classes
    class_weights_ = total_samples / (len(valid_classes) * np.bincount(y_train_labels)[valid_classes])

    if take_weights_log:
        # Apply logarithm to the class weights
        class_weights = np.log(class_weights_ + 1.0)
    else:
        class_weights = class_weights_
    
    # Create a dictionary mapping from class index to weight
    class_weight_dict = dict(zip(valid_classes, class_weights))

    return class_weight_dict, valid_classes


def filter_samples(X_train, y_train, valid_classes, threshold):
    
    """
    Filters the samples of the training set according to the valid classes. A class is considered valid if it meets 
    a certain representation threshold. 

    Parameters
    ----------
    X_train : numpy.ndarray
        The feature variables for the training set. Shape should be (n_samples, n_features).
    y_train : numpy.ndarray
        The one-hot encoded target variables for the training set. Shape should be (n_samples, n_classes).
    valid_classes : list or numpy.ndarray
        A list or array of the classes that are above the representation threshold.
    threshold : float
        The minimum representation threshold for a class to be considered valid. Should be between 0 and 1.

    Returns
    -------
    X_train_filtered : numpy.ndarray
        The filtered feature variables for the training set, with samples belonging to non-valid classes removed. 
        Shape is (n_filtered_samples, n_features).
    y_train_filtered : numpy.ndarray
        The filtered target variables for the training set, with samples belonging to non-valid classes removed. 
        Shape is (n_filtered_samples, n_classes).

    Notes
    -----
    This function only includes samples from classes that are specified as valid. The valid classes should be 
    calculated considering the representation threshold in a prior step, such as with the calculate_class_weights 
    function.
    """
    
    # Convert one-hot encoded y_train to label encoded
    
    y_train_labels = np.argmax(y_train, axis=1)

    # Get a boolean mask where each entry is True if the corresponding sample's label is in valid_classes
    
    valid_samples_mask = np.isin(y_train_labels, valid_classes)

    # Use the mask to filter out the samples from X_train and y_train
    
    X_train_filtered = X_train[valid_samples_mask]
    y_train_filtered = y_train[valid_samples_mask]
    
    print(f'Classes with less than {threshold * len(y_train)} elements will be be removed.')
    print()
    print('==> Done.')

    return X_train_filtered, y_train_filtered
    
    
    
    
    
    