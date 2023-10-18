# This is the main driving script for structuring the code of the project.
# The aim of structuring a project in the way we outline in the way we demonstrate here is that it should be easy to
# read along for people who want to know what you have done. Breaking code into chunks is also a good way to make
# code easier to maintian/extend but also easier for you to collaborate on projects with others.
# What we demonstrate here isn't the best possible way to work on a project, but it should be a simple easy way to get
# started.
# The below is a skeleton of code that can be changed however you want. The arguments to classes and functions here are
# what placeholders, you can change this as you see fit.

print("Loading modules ...")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import numpy as np
import random
import argparse
import time


#get your util functions
from eval import evaluate_my_model, plot_history, ConfusionMatrix
from models import GalaxyZooClassifier
from utils import data_loader, load_file, set_paths, save_hist, save_config, one_hot_encoder_array

print('==> Done.')

def run(config_file, save_model=True, save_history=True, seed=48):
    
    """
    Run the Galaxy Zoo training and evaluation process.
    
    Args:
        config_file (str): Path to the configuration file.
        save_model (bool, optional): Whether to save the trained model. Defaults to True.
        save_history (bool, optional): Whether to save the training history. Defaults to True.
        seed (int, optional): Seed for reproducibility. Defaults to 48.
    """
    
    # Set the seeds so that your code will be reproducible!
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Call the function to save the configuration parameters
    if save_model:
        save_config(config_file)
    
    # Load the config file
    config = load_file(config_file)
    
    # Now you can access your configuration parameters
    paths = config['paths']
    data_preprocessing = config['data_preprocessing']
    model_params = config['model_params']
    
    # Convert the lists to tuples
    data_preprocessing['img_size'] = tuple(data_preprocessing['img_size'])
    model_params['conv_layers'] = [[layer[0], tuple(layer[1])] for layer in model_params['conv_layers']]
    model_params['pool_size'] = tuple(model_params['pool_size'])
    data_augmentation_params = config['data_augmentation_params']
    train_params = config['train_params']
    
    img_directory, labels_path, models_path, plots_path = set_paths(paths)
    
    t_in = time.time()
    print()
    
    # Load the data 

    if model_params['model_type'] == 'base':
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader(images_folder=img_directory, 
                                                                     labels_path=labels_path, **data_preprocessing, 
                                                                     random_seed=seed)
        in_shape = X_train.shape[1:]
        out_shape = y_train.shape[-1]
    else:
        X_train, X_val, X_test, y_train_, y_val_, y_test_ = data_loader(images_folder=img_directory, 
                                                                     labels_path=labels_path, **data_preprocessing, 
                                                                     random_seed=seed)
        in_shape = X_train.shape[1:]
        y_train = [y_train_[:,:3], y_train_[:,3:5], y_train_[:,5:]]
        y_val = [y_val_[:,:3], y_val_[:,3:5], y_val_[:,5:]]
        y_test= [y_test_[:,:3], y_test_[:,3:5], y_test_[:,5:]]
        out_shape = (y_train[0].shape[-1], y_train[1].shape[-1], y_train[2].shape[-1])
    
    # Define the model
    
    galaxy_classifier = GalaxyZooClassifier(input_shape=in_shape, n_classes=out_shape, **model_params, data_augmentation=data_augmentation_params)

    # Train the model
    
    galaxy_classifier.model.summary()
    
    history = galaxy_classifier.fit_model(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **train_params)
    
    print()
    print('==> Done.')

    # Evaluate
    print()
    model = galaxy_classifier.model
    test_data = (X_test, y_test)
    evaluate_my_model(model, test_data)
    
    # Plot history
    
    metric = train_params['metrics']
    
    if not save_history:
        plots_path=None
    
    plot_history(history, metric=metric[0], labels=(config['model_name'], train_params['learning_rate']),folder_path=plots_path)
    
    #if 'regression' not in config['model_name']:
    # Plot Confusion Matrix over the training and test sets
    print('--------------------------------- Calculating CONFUSION MATRIX ------------------------------------')
    train_data = (X_train, y_train)
    ConfusionMatrix(model, data_test = test_data, name = config['model_name']+'_test', folder_path = plots_path)
    ConfusionMatrix(model, data_test = train_data, name = config['model_name']+'_train', folder_path = plots_path)
    print()
    print('==> Done.')
    print()
    print(f'Plots saved in {plots_path}')
    
    t_end = time.time()
    t_elapsed = t_end - t_in
    print()
    print('--------------------------------------------------------------------------')
    print()
    print('Execution time : ', time.strftime('%H:%M:%S', time.gmtime(t_elapsed)))
   
    # Save the model and history 
    
    print('--------------------------------------------------------------------------')
    if save_model is True:
        galaxy_classifier.save_model(models_path=models_path, name=config['model_name'], diagram=True)

    if save_history is True:
        save_hist(history, models_path=models_path, name=config['model_name'])
        
    # Prompt the user to close the plot windows manually
    input("Press Enter to close the plot windows...")
        
    print()
    print('--------------------------------- END ------------------------------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', type=str, help='Path to the configuration file')

    parser.add_argument('--save_model', action='store_true', default=True, help='Whether to save the trained model')
    parser.add_argument('--no-save_model', action='store_false', dest='save_model')

    parser.add_argument('--save_history', action='store_true', default=True, help='Whether to save the training history')
    parser.add_argument('--no-save_history', action='store_false', dest='save_history')

    parser.add_argument('--seed', default=48, type=int, help='Seed for reproducibility')
    
    args = parser.parse_args()
    
    run(args.config, args.save_model, args.save_history, args.seed)