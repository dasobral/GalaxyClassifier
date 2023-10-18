import os, sys, io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml
import pickle

def image_loader(folder, indices=None, crop=False, crop_size=None, img_size=(64, 64), normalize = True, grayscale = False):
    
    """
    Loads all the images contained in the specified folder.

    Parameters
    ----------
    folder : string
        Path to the folder where the images are stored.
    indices : array-like, optional
        Indices of specific images to load. Defaults to None, which loads all images.
    crop : bool, optional
        Whether to crop images before resizing. Defaults to False.
    crop_size : tuple, optional
        Area to crop from the images (left, upper, right, lower). Defaults to None.
    img_size : tuple, optional
        The size to which the original image should be resized. Defaults to (64, 64).
    normalize : bool, optional
        Whether to normalize pixel values to be between 0 and 1. Defaults to True.
    grayscale : bool, optional
        Whether to convert images to grayscale. Defaults to False.

    Returns
    -------
    images : numpy.ndarray
        An array with the images arrays. Shape is (n_images, img_size[0], img_size[1], n_channels), 
        where n_channels is 1 for grayscale and 3 for RGB.

    Raises
    ------
    ValueError
        If crop is True but no crop_size is specified.
    """
    
    filenames = np.array([filename for filename in sorted(os.listdir(folder))])
    if indices is not None:
        filenames = filenames[indices]
    
    color_mode = 'grayscale' if grayscale else 'rgb' 
        
    if crop and crop_size is None:
        raise ValueError("crop_size must be specified when crop=True.")

    if grayscale:
        images = np.empty(shape = (len(filenames), img_size[0], img_size[1], 1))
    else:
        images = np.empty(shape = (len(filenames), img_size[0], img_size[1], 3))
    
    for i, name in enumerate(tqdm(filenames, desc="Loading images", ncols=75, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')):
        img_path = os.path.join(folder, name)
        img = keras.utils.load_img(img_path, color_mode=color_mode)
        
        if crop:
            width, height = img.size
            left = (width - crop_size) // 2
            upper = (height - crop_size) // 2
            right = left + crop_size
            lower = upper + crop_size
            img = img.crop((left, upper, right, lower))

        img = img.resize(size = img_size)
        img_ = keras.preprocessing.image.img_to_array(img)
        
        if grayscale:
            img_ = np.squeeze(img_, axis=-1)  # Remove the redundant channel
            img_ = np.expand_dims(img_, axis=-1)  # Add the channel dimension back
            
        if normalize:
            img_ = img_ / 255.0
             
        images[i] = img_
        
    return images

def labels_loader(folder, task = [1, 2], min=0.8, one_hot_enc = True):
    
    """
    Loads labels from the specified file and performs optional transformations.

    Parameters
    ----------
    folder : string
        Path to the file where the labels are stored.
    task : list, optional
        Selects the relevant labels according to the Task. Default value [1, 2] gives the labels dataset.
    min : float, optional
        The minimum value to keep a label. Rows with all label values below this threshold are dropped. Defaults to 0.8.
    one_hot_enc : bool, optional
        Whether to perform one-hot encoding on the labels. Defaults to True.

    Returns
    -------
    final_df : pandas.DataFrame
        A DataFrame with the labels arrays.

    Notes
    -----
    The function assumes that the labels are stored in a CSV file with a specific structure. Make sure that the 
    task parameter matches the structure of your file.
    """
    
    data = pd.read_csv(folder)
        
    if task == 1:
        data_ = data.iloc[:,:4]
        data_df = data_[data_.iloc[:,1:].max(axis=1) >= min]
        if one_hot_enc == True:
            data_df_enc = one_hot_encoder(data_df.iloc[:, 1:])
            final_df = pd.concat([data_df['GalaxyID'], data_df_enc], axis=1)
        else:
            final_df=data_df.copy()
    else:
        data_ = data[['GalaxyID', 'Class1.1', 'Class1.2', 'Class1.3', 'Class2.1', 'Class2.2', 'Class7.1', 'Class7.2', 'Class7.3']]
        data_df = data_[data_.iloc[:,1:].max(axis=1) >= min]
        if one_hot_enc == True:
            final_df = hierarchy_one_hot(data_df, min=0.5)
        else:
            final_df=data_df.copy()
        
    return final_df

def data_loader(images_folder, labels_path, task, min=0.8, one_hot_labels = False, crop=False, crop_size=None, img_size=(64, 64), normalize=True, grayscale = 0, training_size = 0.8, test_size = None, random_seed=48):
    
    """
    Load and split the data into training and validation (and optionally test) sets.

    Parameters
    ----------
    images_folder : str
        Path to the folder where the images are stored.
    labels_path : str
        Path to the file where the labels are stored.
    task : int
        Task to work on.
    min : float, optional
        The minimum value to keep a label. Rows with all label values below this threshold are dropped. Defaults to 0.8.
    one_hot_labels : bool, optional
        Whether to perform one-hot encoding on the labels. Defaults to False.
    crop : bool, optional
        Whether to crop images before resizing. Defaults to False.
    crop_size : tuple, optional
        Area to crop from the images (left, upper, right, lower). Defaults to None.
    img_size : tuple, optional
        The size to which the original image should be resized. Defaults to (64, 64).
    normalize : bool, optional
        Whether to normalize pixel values to be between 0 and 1. Defaults to True.
    grayscale : int, optional
        Whether to convert images to grayscale. 1 means grayscale, 0 means RGB. Defaults to 0 (RGB).
    training_size : float, optional
        Proportion of the dataset to include in the training split. Default is 0.8.
    test_size : float, optional
        If not None, proportion of the training set to include in the test split. Default is None.
    random_seed : int, optional
        Seed used by the random number generator for reproducibility. Default is 48.

    Returns
    -------
    tuple
        Contains the training data (X_train, y_train), validation data (X_val, y_val), and if test_size is not None, 
        test data (X_test, y_test) too.

    Notes
    -----
    The function assumes that the labels are stored in a CSV file with a specific structure. 
    Make sure that the task parameter matches the structure of your file.
    """

    gray_scale = True if grayscale == 1 else False 
    
    if task == 1:
        labels_df = labels_loader(labels_path, task=task, min=min, one_hot_enc=one_hot_labels)
        labels_indices = labels_df.index
        images = image_loader(folder=images_folder, indices=labels_indices, crop=crop, crop_size=crop_size, img_size=img_size, normalize=normalize, grayscale=gray_scale)
       
    else:
        labels_df = labels_loader(labels_path, task=task, min=min, one_hot_enc=one_hot_labels)
        labels_indices = labels_df.index
        images = image_loader(images_folder, indices=labels_indices, crop=crop, crop_size=crop_size, img_size=img_size, normalize=normalize, grayscale=gray_scale)
        
    
    X = images
    labels_ = labels_df.copy()
    labels_ = labels_.drop(labels_.columns[0], axis=1)
    labels_array = labels_.to_numpy()
    
    X_train, X_val, y_train, y_val = train_test_split(X, labels_array, train_size=training_size, random_state=random_seed)
    
    if test_size == None:
        return X_train, X_val, y_train, y_val
    else:
        X_train_, X_test, y_train_, y_test = train_test_split(X_train, y_train, test_size=test_size, random_state=random_seed)
        
        return X_train_, X_val, X_test, y_train_, y_val, y_test

def image_plotter(images, names, ncols, plot_size=(12,4), spacing=(0.025, 0.025), title=None, folder_path=None):
    
    """
    Plot and optionally save an array of images in a grid layout.

    Parameters
    ----------
    images : array
        The array that contains the images to be plotted.
    names : list
        List of image names/titles to be displayed above each image.
    ncols : int
        The number of columns in the plot grid.
    plot_size : tuple, optional
        The overall size of the entire plot. Defaults to (32,20).
    spacing : tuple, optional
        The spacing between images in the plot, defined as (vertical space, horizontal space). Defaults to (0.025, 0.025).
    folder_path : str, optional
        If provided, the plot will be saved as a PDF file in the specified folder. The folder is created if it does not exist. Default is None.

    Returns
    -------
    None
        This function does not return a value. It shows the plot in the output.

    Notes
    -----
    The images are plotted in a grid layout with a specified number of columns (ncols). The number of rows is determined
    based on the total number of images and ncols. Each image is displayed with its respective title from 'names'.
    """
        
    ncols = len(images)
    nrows = len(images) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=plot_size)
    for idx, ax in enumerate(axs.flat):
        ax.imshow(images[idx])
        ax.set_title('ID: '+names[idx])  # Add the image title
        ax.axis('off')
        
    plt.subplots_adjust(hspace=spacing[0], wspace=spacing[1], top=0.99, bottom=0.01, left=0.01, right=0.99)  # Adjust the spacing between subplots
    
    # Save the plot if a folder_path is provided
    if folder_path is not None:
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(os.path.join(folder_path, title+'.pdf'), format='pdf')    
    
    plt.show()

def load_file(file):
    
    """
    Loads a YAML configuration file.

    Parameters
    ----------
    file : str
        The path to the YAML configuration file to load.

    Returns
    -------
    dict
        A dictionary containing the configuration options loaded from the file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.

    Notes
    -----
    The function raises an exception if the file does not exist. The function assumes the file content is in the YAML format. 
    """
    
    if not os.path.isfile(file):
        raise FileNotFoundError(f"Configuration file '{file}' not found. It should be in the same directory as your script.")
    
    with open(file, 'r') as file_obj:
        config = yaml.safe_load(file_obj)
        
    return config

def set_paths(paths):
    
    """
    Sets the paths for loading data and saving outputs (models and plots).

    Parameters
    ----------
    paths : dict
        A dictionary containing the paths. It should include the following keys:
        'images_path' - Path to the directory where the image data is stored.
        'labels_path' - Path to the file where the labels are stored.
        'models_path' - Path to the directory where models should be saved.
        'plots_path' - Path to the directory where plots should be saved.

    Returns
    -------
    tuple
        A tuple containing four strings: the path to the images, the path to the labels, the path to the models directory,
        and the path to the plots directory.

    Notes
    -----
    The function creates the directories for saving models and plots if they do not already exist.
    """
    
    # Set the location where you can load data from
    img_directory = paths['images_path']
    labels_path = paths['labels_path']
    
    # Set the location where outputs are saved
    models_path = paths['models_path']
    os.makedirs(models_path, exist_ok=True)
    plots_path = paths['plots_path']
    os.makedirs(plots_path, exist_ok=True)
    
    return img_directory, labels_path, models_path, plots_path

def save_config(config_file):
    
    """
    Prints the configuration parameters and saves them into a log file.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.

    Notes
    -----
    This function captures the current state of the standard output (stdout), redirects it to a string IO object,
    and restores it after the operation. The captured output (configuration parameters) is written into a text file.
    
    The 'sort_keys' parameter in the yaml.dump() method is set to False. This means the order of keys in the original
    YAML file is preserved in the output, rather than being sorted in default Python's lexicographical order.
    The function creates a directory named 'logs' if it does not already exist to store the log files.

    """
    
    # Create a stream to capture the printed output
    stdout_backup = sys.stdout
    sys.stdout = io.StringIO()

    # Load the config file
    # Print the configuration parameters
    print('\n----- CONFIGURATION PARAMETERS -----\n')
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
        yaml.dump(config_data, sys.stdout, sort_keys=False)

    # Restore the standard output
    output = sys.stdout.getvalue()
    sys.stdout.close()
    sys.stdout = stdout_backup

    # Create a logs directory if it doesn't exist
    logs_dir = 'logs/'
    os.makedirs(logs_dir, exist_ok=True)

    # Save the log to a text file
    log_file = f"{logs_dir}/log_{config_data['model_name']}.txt"
    with open(log_file, 'w') as f:
        f.write(output)
        
def save_hist(history, models_path, name):
    
    """
    Saves the history of a model's training into a pickle file.

    Parameters
    ----------
    history : History object
        The history object generated by the training process of a model.
    models_path : str
        The path to the directory where the history file will be saved.
    name : str
        The name to be given to the saved history file.

    Notes
    -----
    The history object typically includes useful data like loss and accuracy metrics recorded at each epoch
    during the training process. It is saved using the pickle module, which allows for serialization and de-serialization
    of Python object structures. 
    """
    
    history_path = os.path.join(models_path, 'History_'+name)
    with open(history_path, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    print()
    print(f'History saved in {history_path}')
    
def one_hot_encoder(df):
    
    """
    Converts a DataFrame to one-hot encoding.

    This function takes a DataFrame and converts it to one-hot encoded form. This means that each row is a vector
    where only the element corresponding to the category of that sample is set to 1 and all other elements are 0.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to be converted to one-hot encoding. 
        It is expected that the DataFrame is numeric and that the column with the highest value in each row 
        corresponds to the category of that sample.

    Returns
    -------
    one_hot_df : DataFrame
        A DataFrame that is the one-hot encoded version of the input df. 
        It has the same shape as the input df, with values replaced by 0s and 1s.
    """
    
    one_hot_df = df.copy()
    max_indices = one_hot_df.idxmax(axis=1)
    one_hot_df[:] = 0.
    for idx, col_name in zip(one_hot_df.index, max_indices):
        one_hot_df.at[idx, col_name] = 1.
        
    return one_hot_df

def hierarchy_one_hot(df, min = 0.5):
    
    """
    Converts a DataFrame to one-hot encoding based on the maximum value in each hierarchical group.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be converted to one-hot encoding. 
        It should contain columns for each class and sub-class.
    threshold : float, optional
        The threshold value used for the one-hot encoding of Class1. Defaults to 0.5.

    Returns
    -------
    df_one_hot : pandas.DataFrame
        A DataFrame that is the one-hot encoded version of the input df based on the max value in each group.
    """
    
    df_one_hot = df.copy()

    # Binary encoding of Class1
    df_one_hot[['Class1.1', 'Class1.2', 'Class1.3']] = (df_one_hot[['Class1.1', 'Class1.2', 'Class1.3']] > min).astype(float)

    # Binary encoding of Class2, conditioned on Class1.2
    df_one_hot['Class2.1'] = (df_one_hot[['Class2.1', 'Class2.2']].idxmax(axis=1) == 'Class2.1').astype(float)
    df_one_hot['Class2.2'] = (df_one_hot[['Class2.1', 'Class2.2']].idxmax(axis=1) == 'Class2.2').astype(float)
    df_one_hot.loc[df_one_hot['Class1.2'] == 0, ['Class2.1', 'Class2.2']] = 0

    # Binary encoding of Class7, conditioned on Class1.1
    df_one_hot['Class7.1'] = (df_one_hot[['Class7.1', 'Class7.2', 'Class7.3']].idxmax(axis=1) == 'Class7.1').astype(float)
    df_one_hot['Class7.2'] = (df_one_hot[['Class7.1', 'Class7.2', 'Class7.3']].idxmax(axis=1) == 'Class7.2').astype(float)
    df_one_hot['Class7.3'] = (df_one_hot[['Class7.1', 'Class7.2', 'Class7.3']].idxmax(axis=1) == 'Class7.3').astype(float)
    df_one_hot.loc[df_one_hot['Class1.1'] == 0, ['Class7.1', 'Class7.2', 'Class7.3']] = 0

    return df_one_hot

def one_hot_encoder_array(array):
    
    """
    Converts an array to one-hot encoding.

    This function takes a 2D numpy array and converts it to one-hot encoded form. This means that each row is a vector
    where only the element corresponding to the category of that sample is set to 1 and all other elements are 0.

    Parameters
    ----------
    array : ndarray
        The 2D numpy array to be converted to one-hot encoding. 
        It is expected that the array is numeric and that the column with the highest value in each row 
        corresponds to the category of that sample.

    Returns
    -------
    new_array : ndarray
        A 2D numpy array that is the one-hot encoded version of the input array. 
        It has the same shape as the input array, with values replaced by 0s and 1s.
    """
    
    max_indices = np.argmax(array, axis=1)
    new_array = np.zeros_like(array)
    new_array[np.arange(len(array)), max_indices] = 1
    return new_array

def data_inspector(labels):
    # Create a new DataFrame to store the assigned classes
    assigned_classes = pd.DataFrame(index=labels.index)

    # Assign each sample to the Class1 subclass with the highest probability
    assigned_classes['Class1'] = labels[['Class1.1', 'Class1.2', 'Class1.3']].idxmax(axis=1)

    # For those assigned to Class1.2, assign them to the Class2 subclass with the highest probability
    mask_class1_2 = assigned_classes['Class1'] == 'Class1.2'
    assigned_classes.loc[mask_class1_2, 'Class2'] = labels.loc[mask_class1_2, ['Class2.1', 'Class2.2']].idxmax(axis=1)

    # For those assigned to Class1.1, assign them to the Class7 subclass with the highest probability
    mask_class1_1 = assigned_classes['Class1'] == 'Class1.1'
    assigned_classes.loc[mask_class1_1, 'Class7'] = labels.loc[mask_class1_1, ['Class7.1', 'Class7.2', 'Class7.3']].idxmax(axis=1)

    class1_dict = {'Class1.1': 'Smooth', 'Class1.2': 'Features/Disk', 'Class1.3': 'Star/Artifact'}
    class2_dict = {'Class2.1': 'Normal Disk', 'Class2.2': 'Edge-on viewed Disk'}
    class7_dict = {'Class7.1': 'Round', 'Class7.2': 'Eliptical', 'Class7.3': 'Cigar'}

    class1_counts = assigned_classes['Class1'].value_counts(sort=False).to_dict()
    class2_counts = assigned_classes[assigned_classes['Class1']=='Class1.2']['Class2'].value_counts(sort=False).to_dict()
    class7_counts = assigned_classes[assigned_classes['Class1']=='Class1.1']['Class7'].value_counts(sort=False).to_dict()

    print("Counts of Class1 Subclasses:")
    for class_label, count in sorted(class1_counts.items()):
        print(f'{class_label} -> {class1_dict[class_label]}: {count} samples')
    print()

    print("Counts of Class2 Subclasses among those assigned to Class1.2:")
    for class_label, count in sorted(class2_counts.items()):
        print(f'{class_label} -> {class2_dict[class_label]}: {count} samples')
    print()

    print("Counts of Class7 Subclasses among those assigned to Class1.1:")
    for class_label, count in sorted(class7_counts.items()):
        print(f'{class_label} -> {class7_dict[class_label]}: {count} samples')