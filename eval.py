# Here you can add some evaluation specific functions

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_my_model(model, test_data, set_name = 'testing'):
    
    """
    Evaluates a trained model using the provided testing data.

    Parameters
    ----------
    model : keras.Model
        The trained model to be evaluated.
    test_data : list
        A list containing the testing data. The first element of the list 
        is expected to be the input data (features), and the second element 
        is expected to be the labels (target).
    set_name : str, optional
        A string specifying the name of the testing dataset. It is only used for 
        printing purposes to inform the user about the dataset being used for evaluation. 
        The default value is 'testing'.

    Notes
    -----
    This function assumes that the `test_data` list contains exactly two elements. 
    The first element is expected to be the input data (features), 
    and the second element is expected to be the labels (target). 

    This function uses the `evaluate` method of the keras.Model class to evaluate the model. 
    The `evaluate` method returns a list of metrics values. The metrics are specified 
    during the compilation of the model.

    After the model is evaluated, this function prints the name of each metric 
    and its corresponding value.
    """
    
    # Evaluate the model
    
    print('----- EVALUATING MODEL -----')
    print(f'Evaluating the model over the {set_name} set')

    score = model.evaluate(test_data[0], test_data[1])
    
    print()
    for i, name in enumerate(model.metrics_names):
        print('Test '+name+f': {score[i]}')
        
def plot_history(history, metric, labels, size=(18, 6), folder_path=None):
    
    """
    Plots the total training and validation losses and specified metrics from a model's history.
    
    Parameters
    ----------
    history : keras.callbacks.History object
        The history object generated by the training method `model.fit()`.
    metric : str
        The base name of the metric(s) to be plotted. The function will plot all metrics 
        that contain this string in their name.
    labels : tuple
        A tuple containing labels for the generated plot. The tuple is expected 
        to have two elements - `model_name` and `learning_rate`.
    size : tuple, optional
        A tuple specifying the size of the generated plot. The default size is (12, 5).
    folder_path : str, optional
        A string specifying the path to the folder where the plot should be saved. 
        If this is set to None, the plot is not saved. The default value is None.
        
    Returns
    -------
    None
        
    Notes
    -----
    This function generates two plots - one for the total loss and one for the specified metric(s).
    Each plot shows the values of the respective quantity for both the training and 
    validation data across the epochs of training. If the model has multiple outputs,
    the function plots the specified metric(s) for each output on the right subplot.
    
    If `folder_path` is not None, this function saves the generated plot in that folder 
    as a pdf file. The file is named as `<model_name>_<learning_rate>.pdf`. It is assumed 
    that `labels` is a tuple containing exactly two elements - `model_name` and `learning_rate`.

    This function uses matplotlib.pyplot for generating the plots. This must be installed 
    in your environment in order to use this function.
    
    """
    
    # Convert the metric to string if it's a function
    if callable(metric):
        metric = metric.__name__
        
    #aspect_ratio = fig_width / fig_height
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=size)
    #plt.gca().set_aspect(aspect_ratio)
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Train', linestyle='--')
    ax1.plot(history.history['val_loss'], label='Validation')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    
    # Plot metrics
    for key in history.history.keys():
        if metric in key and 'loss' not in key:  # Ignore keys that contain 'loss'
            if 'val_' in key:  # If 'val_' is in the key, this is a validation metric
                ax2.plot(history.history[key], label=key, linestyle='-')  # Plot with a solid line
            else:  # Otherwise, this is a training metric
                ax2.plot(history.history[key], label=key, linestyle='--')  # Plot with a dashed line

    ax2.set_title('Model Metrics')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel(metric)
    ax2.legend(loc= 'best')
    
    fig.subplots_adjust(wspace=0.15, left=0.05, right=0.95)
    
    if folder_path is not None:
        model_name, learning_rate = labels
        os.makedirs(folder_path, exist_ok=True)
        file_name = f"{folder_path}/{model_name}_{learning_rate}.pdf"
        plt.savefig(file_name, format="pdf")
    
    plt.pause(0.001)
    #plt.show(block=False)
    
def ConfusionMatrix(model, data_test, name, folder_path=None):
    
    """
    Generates and plots a confusion matrix for the given model and test data.

    Parameters
    ----------
    model : keras Model object
        The model for which the confusion matrix is to be generated.
    data_test : tuple
        A tuple containing the test data. The tuple is expected to contain 
        two numpy arrays - `X_test` and `y_test`, where `X_test` is the 
        input data and `y_test` are the true labels.
        In the case of multi-output models, `y_test` should be a list 
        containing the `y_test[i]` arrays for each output.
    name : str
        A string to be used in the title of the generated plot.
    folder_path : str, optional
        A string specifying the path to the folder where the plot should 
        be saved. If this is set to None, the plot is not saved. The default 
        value is None.

    Returns
    -------
    None

    Notes
    -----
    This function generates a confusion matrix for the given model and test 
    data and plots it using seaborn. It first uses the model to predict labels 
    for the test data. It then compares these predicted labels with the true 
    labels to generate the confusion matrix.

    The function generates a plot showing the confusion matrix and saves it 
    in the specified folder if `folder_path` is not None. The plot is saved 
    as a pdf file named `'ConfusionMatrix_<name>.pdf'`.

    If `y_test` is a list for multi-output models, the function creates a subplot 
    grid based on the number of outputs. Each subplot represents the confusion 
    matrix for a specific output. The overall title for the plot indicates the 
    name of the model and the type of output. The size of each subplot can be 
    adjusted by modifying the `subplot_size` parameter in the function.

    If `y_test` is a single numpy array, the function generates a single plot 
    for the confusion matrix.

    The seaborn and matplotlib libraries must be installed in your environment 
    to use this function.
    """
    
    X_test, y_test = data_test
    y_pred = model.predict(X_test)
    
    # Check if y_test is a list for multi-output models
    if isinstance(y_test, list):
        # Predict labels for each output
        
        num_outputs = len(y_test)

        # Calculate the size of each subplot based on the number of outputs
        subplot_size = 6  # Define the desired size of each subplot
        fig_width = subplot_size * num_outputs + 10
        fig_height = subplot_size + 2
        
        #aspect_ratio = fig_width / fig_height 

        # Create a subplot grid with one row and num_outputs columns
        fig, axes = plt.subplots(1, num_outputs, figsize=(fig_width, fig_height))
        
        Class7_mask = ( np.argmax(y_pred[0], axis=1) == 0 )
        Class2_mask = ( np.argmax(y_pred[0], axis=1) == 1 )
        
        y_test_ = [y_test[0],  y_test[1][Class2_mask], y_test[2][Class7_mask]]
        y_pred_ = [y_pred[0], y_pred[1][Class2_mask], y_pred[2][Class7_mask]]
    
        # Calculate confusion matrix for each output separately
        for i, ax in enumerate(axes.flatten()):
            y_test_i = np.argmax(y_test_[i], axis=1)
            y_pred_i = np.argmax(y_pred_[i], axis=1)
            cm = confusion_matrix(y_test_i, y_pred_i)

            # Plot the confusion matrix in the subplot
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            ax.set_title(f'Classes {i+1}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Truth')
            
        # Adjust the size of each subplot
        fig.subplots_adjust(wspace=0.05, left=0.025, right=0.9990)

        # Set the overall title for the plot
        fig.suptitle(f'Confusion Matrices {name}', fontsize=14)

        if folder_path is not None:
            # Ensure that the folder exists
            os.makedirs(folder_path, exist_ok=True)

            # Save the plot as a pdf file
            file_name = f"{folder_path}/{'ConfusionMatrix'}_{name}.pdf"
            plt.savefig(file_name, format="pdf")
            
        plt.pause(0.001)
        #plt.show(block=False)
    else:
        # Convert predicted probabilities to classes
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Convert true labels to classes
        y_test_classes = np.argmax(y_test, axis=1)

        # Calculate confusion matrix for all classes
        cm = confusion_matrix(y_test_classes, y_pred_classes)

        # Create a single plot for the confusion matrix
        
        fig_width = 10
        fig_height = 6
        
        #aspect_ratio = fig_width / fig_height        
        
        plt.figure(figsize=(fig_width, fig_height))
        #plt.gca().set_aspect(aspect_ratio)

        sns.heatmap(cm, annot=True, fmt='d')
        plt.title(f'Confusion Matrix {name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        if folder_path is not None:
            # Ensure that the folder exists
            os.makedirs(folder_path, exist_ok=True)

            # Save the plot as a pdf file
            file_name = f"{folder_path}/{'ConfusionMatrix'}_{name}.pdf"
            plt.savefig(file_name, format="pdf")
        plt.pause(0.001)
        #plt.show(block=False)
        
def ConfusionMatrix_(model, data_test, name, folder_path):
    
    """
    Generates and plots a confusion matrix for the given model and test data.
    
    Parameters
    ----------
    model : keras Model object
        The model for which the confusion matrix is to be generated.
    data_test : tuple
        A tuple containing the test data. The tuple is expected to contain 
        two numpy arrays - `X_test` and `y_test`, where `X_test` is the 
        input data and `y_test` are the true labels.
    name : str
        A string to be used in the title of the generated plot.
    folder_path : str, optional
        A string specifying the path to the folder where the plot should 
        be saved. If this is set to None, the plot is not saved. The default 
        value is None.
        
    Returns
    -------
    None
        
    Notes
    -----
    This function generates a confusion matrix for the given model and test 
    data and plots it using seaborn. It first uses the model to predict labels 
    for the test data. It then compares these predicted labels with the true 
    labels to generate the confusion matrix.

    The function generates a plot showing the confusion matrix and saves it 
    in the specified folder if `folder_path` is not None. The plot is saved 
    as a pdf file named `'ConfusionMatrix_<name>.pdf'`.
    
    The seaborn library must be installed in your environment to use this function.
    """
    
    # Calculate and print the Confusion Matrix
    
    X_test, y_test = data_test
    y_pred = model.predict(X_test)
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    
    fig_width = 10
    fig_height = 6
        
    #aspect_ratio = fig_width / fig_height        
        
    plt.figure(figsize=(fig_width, fig_height))
    #plt.gca().set_aspect(aspect_ratio)
    
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'Confusion Matrix {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    
    if folder_path is not None:
    
        # Ensure that the folder exists
        os.makedirs(folder_path, exist_ok=True)
        
        # Save the plot as a pdf file
        file_name = f"{folder_path}/{'ConfusionMatrix'}_{name}.pdf"
        plt.savefig(file_name, format="pdf")
        
    plt.pause(0.001)
    #plt.show(block=False)
    
def error_analysis(model, data_test):
    
    """
    Performs error analysis on the model's predictions and returns indices of the misclassified data.

    Parameters
    ----------
    model : keras Model object
        The trained model which predictions are to be analyzed.
    data_test : tuple
        A tuple containing the test data. The tuple is expected to contain 
        two numpy arrays - `X_test` and `y_test`, where `X_test` is the 
        input data and `y_test` are the true labels.

    Returns
    -------
    wrong_indices : numpy.ndarray
        An array containing the indices of the test data that were misclassified.
    y_test_labels : numpy.ndarray
        An array of the true labels.
    y_pred_labels : numpy.ndarray
        An array of the predicted labels.

    Notes
    -----
    This function makes predictions on the test data using the provided model, 
    then compares the predicted labels with the true labels to identify the indices 
    where the model's predictions were incorrect. These indices are then returned 
    along with the true and predicted labels for further analysis.
    """
    
    X_test, y_test = data_test
    y_pred = model.predict(X_test)
    
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Get the indices of the examples that your model got wrong
    wrong_indices = np.where(y_test_labels != y_pred_labels)[0]

    # Print the total number of errors
    print(f'Total errors: {len(wrong_indices)}')
    
    return wrong_indices, y_test_labels, y_pred_labels 

def compare_plot(model, data_test, label_dict, folder_path, num_img = 6):
    
    """
    Plots and compares a number of randomly selected misclassified test images along with their predicted and true labels.
    
    Parameters
    ----------
    model : keras Model object
        The trained model which predictions are to be analyzed.
    data_test : tuple
        A tuple containing the test data. The tuple is expected to contain 
        two numpy arrays - `X_test` and `y_test`, where `X_test` is the 
        input data and `y_test` are the true labels.
    label_dict : dict
        Dictionary mapping the labels represented as integers to their actual values.
    folder_path : str
        Path to the folder where the generated plot will be saved.
    num_img : int, optional
        Number of misclassified images to be displayed. Default is 6.

    Notes
    -----
    This function selects a number of images that were misclassified by the model,
    then generates a plot displaying these images along with their true and predicted labels.
    The plot is saved to a specified location if `folder_path` is not None.
    """
    
    X_test, _ = data_test
    wrong_indices, y_test_labels, y_pred_labels = error_analysis(model, data_test)
    
    # Randomly select a few indices from the list of wrong ones
    random_indices = np.random.choice(wrong_indices, size=num_img, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for idx, ax in zip(random_indices, axes.ravel()):
    
        # Display image
        ax.imshow(X_test[idx])  # adjust as needed if your images are not grayscale
    
        true_label = label_dict[y_test_labels[idx]]
        pred_label = label_dict[y_pred_labels[idx]]
    
        ax.set_title(f'True: {true_label}\nPred: {pred_label}')
        ax.axis('off')

    plt.tight_layout()
    
    if folder_path is not None:
    
        # Ensure that the folder exists
        os.makedirs(folder_path, exist_ok=True)
        
        # Save the plot as a pdf file
        file_name = f"{folder_path}/{'MisclassifiedImages'}.pdf"
        plt.savefig(file_name, format="pdf")
    
    plt.pause(0.001)
    #plt.show(block=False)     