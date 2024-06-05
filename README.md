# Transformer Interpretability

This project aims to explore the use of transformer models for learning SQL programs. It includes the implementation of various SQL scenarios, dataset generation, and training of transformer models to understand and perform SQL tasks.

## Directory Structure

- `experiments.ipynb` - Jupyter notebook with RASP-modeled SQL scenarios.
- `src` - Directory containing code for learning transformer programs.
  - `data_utils` - Contains code for generating datasets for SQL learning.
  - `run.py` - Script to run the learning process. Configuration changes are required in `launch.json`.
- `experiment_results` - Directory where the results of training the discrete transformer models for SQL scenarios are stored.

## Setup Instructions

1. **Clone the repository**
    ```bash
    git clone https://github.com/your-repo/project-name.git
    cd project-name
    ```

2. **Install the required packages**
    ```bash
    pip install -r requirements.txt
    pip install . (installs local code as package)
    ```

3. **Generate Datasets**
   - Navigate to the `src/data_utils` directory.
   - Run the dataset generation scripts to create the necessary datasets for training.
    ```bash
    python generate_datasets.py
    ```

4. **Run the Learning Process**
   - Modify the configuration settings in `launch.json` as needed.
   - Execute the `run.py` script to start the learning process.
    ```bash
    python src/run.py
    ```

## Usage

- **Experiments Notebook**: The `experiments.ipynb` notebook contains various SQL scenarios modeled using RASP. Open and run this notebook to explore the scenarios.
- **Training Models**: Use the `src/run.py` script to train the transformer models. Ensure to configure the settings in `launch.json` before running the script.
- **Results**: After training, the results will be available in the `experiment_results` directory.

## Results

The results of the training process, including model performance and evaluations, are stored in the `experiment_results` directory. Refer to this directory for detailed insights and analysis of the trained models.
