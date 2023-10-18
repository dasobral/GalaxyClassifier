
# Galaxy Zoo Classifier

## Overview
The Galaxy Zoo Classifier is a framework to build, compile and train deep learning models designed to classify galaxies based on their images. It uses convolutional neural network (CNN) architectures to learn and predict the classes of galaxies.

## Setup
Before using the Galaxy Zoo Classifier, it is recommended to set up a Conda environment with TensorFlow. This ensures that the necessary dependencies are isolated and managed efficiently. Here are the steps to set up the environment:

1. First, install Anaconda or Miniconda if you haven't already. Anaconda is a distribution of Python and R for scientific computing, while Miniconda is a smaller, "minimal installer" version that only includes Conda and its dependencies.

2. Once you have Anaconda or Miniconda installed, create a new Conda environment. You can name it whatever you want, but in this example, we'll call it `galaxy_env`. To create the environment, open your terminal and run the following command:
   ```shell
   conda create --name galaxy_env
   ```

3. Activate the newly created environment by running:
   ```shell
   conda activate galaxy_env
   ```

4. Install TensorFlow in the Conda environment. You might prefer to choose the latest stable version:
   ```shell
   conda install -c conda-forge tensorflow
   ```

5. Now that TensorFlow is installed, clone the Galaxy Zoo Classifier project into your desired directory and navigate to the project's root directory in the terminal.

6. Install the other necessary dependencies by running the following command. You might need to modify the requirements.txt file to specify the version you want to install:
   ```shell
   pip install -r requirements.txt
   ```

## Usage
To run the Galaxy Zoo Classifier without saving the model or training history, you can use the following instructions:

1. Prepare your data:
   - Ensure that your galaxy images are stored in the directory specified by `paths.images_path`.
   - Create a YAML configuration file (`config.yaml`) and update it with your specific paths and parameters. An example configuration file is provided.
   - Modify the configuration file according to your needs, including data preprocessing settings, model parameters, and training parameters.

2. Run the code by executing the `run.py` script with the configuration file as a command-line argument:
   ```shell
   python run.py --config config.yaml --no-save_model --no-save_history --seed 48
   ```

3. The code will load the configuration, preprocess the data, build the model, train it, evaluate its performance, and generate plots and a confusion matrix. However, the model and training history will not be saved.

Note that the `--no-save_model` and `--no-save_history` flags are used to disable the saving of the model and training history, respectively. By default, both saving options are enabled (`--save_model` and `--save_history`), so you need to explicitly specify the flags to disable them. The `--seed` flag is used to set the random seed for all the randomized processes. The value 48 is the default value, being the value that must be chosen for reproducibility of the results.

## Configuration
The configuration file (`config.yaml`) contains the following sections. For more information about the role of each configuration parameter, please see the code documentation. A few  important notes:

1. The `task` parameter defines the task to solve. We have only considered the first and second tasks of the proposed project, with values `1` and `2` respectively. 

2. One has to set by hand the parameter `one_hot_labels` to `True` for each task if desired.

3. The `conv_layers` and `pool_size` must be passed as a list of lists. They are transformed to tuples in the `run.py` file. Being the configuration file `.yaml`, it does not recognize tuples by default.

4. The parameters which require a number as input such as `crop_size`, `dropout_rate`, `early_stop_patience` and so on must be passed as `null` if not wanted. The same applies to the `data_augmentation_params` dictionary. The code was written to take those as `None` if not desired, but `.yaml` files require `null` which then is transformed to `None` in Python.

5. If `class_weights` set to `True` the class weights are calculated. However, if you don't use the custom losses the weights are not applied. 

```yaml
paths:
  images_path: 'galaxy-zoo_data/images_training_rev1/'
  labels_path: 'galaxy-zoo_data/training_solutions_rev1.csv'
  plots_path: 'metrics_plots/'
  models_path: 'saved_models/'

data_preprocessing:
  task: 2
  min: 0.5
  one_hot_labels: False
  crop: True
  crop_size: 256
  img_size: [64, 64]
  normalize: True
  grayscale: False
  training_size: 0.80
  test_size: 0.2

model_name: 'Base_ModelII_regression'

model_params:
  model_type: 'base'
  conv_layers: [[32, [3,3]], [64, [3,3]], [128, [3,3]]]
  dense_units: [256, 128]
  batch_normalization: False
  activation: 'relu'
  pool_size: [2, 2]
  flattening: 'Flatten'
  class_weights: False
  out_activation: 'sigmoid'
  dropout_rate: 0.25
  max_out: True
  early_stop_patience: 15
  monitor: 'val_loss'

data_augmentation_params:
  rotation_range: 90
  width_shift_range: 0.01
  height_shift_range: 0.01
  horizontal_flip: True
  vertical_flip: True
  shear_range: 0.015
  zoom_range: 0.15

train_params:
  learning_rate: 0.001
  loss_function: 'mean_squared_error'
  metrics: ['mse', 'accuracy']
  batch_size: 100
  epochs: 150
  threshold: 0.0
  take_weights_log: True
```

## Customization
Feel free to customize the code to fit your specific needs. You can modify the model architecture, experiment with different data preprocessing techniques, or adjust the training parameters. Additionally, you can extend the code by adding new functionality or implementing advanced features.

## Dependencies
The Galaxy Zoo Classifier relies on the following dependencies:
- TensorFlow
- NumPy
- argparse

Ensure that these dependencies are installed before running the code.

## References
- Example configuration file: `config.yaml`
- `run.py`: Main script for running the code
- `eval.py`: Evaluation functions for model performance
- `models.py`: Definition of the GalaxyZooClassifier model
- `utils.py`: Utility functions for data loading and preprocessing

## License
The Galaxy Zoo Classifier is open source and distributed under the [MIT License](LICENSE).

