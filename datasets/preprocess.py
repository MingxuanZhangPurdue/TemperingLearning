import argparse
import os
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ucimlrepo import fetch_ucirepo

name_to_id_dict = {
    "wine_quality": 186,
    "student_performance": 320,
    "abalone": 1,
    "automobile": 10,
    "auto_mpg": 9,
    "bike_sharing": 275
}

def parse_arguments():

    parser = argparse.ArgumentParser(description='Preprocess UCI dataset')

    parser.add_argument(
        '--dataset_name', 
        type=str, 
        default='wine_quality',
        choices=name_to_id_dict.keys(),
        help='Name of the dataset to preprocess'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--test_size', 
        type=float, 
        default=0.2,
        help='Proportion of the dataset to include in the test split'
    )

    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./datasets/',
        help='Directory to save the preprocessed data'
    )
    return parser.parse_args()


def preprocess_dataset(dataset_name, test_size=0.2, random_state=42):

    # Load the dataset
    dataset = fetch_ucirepo(id=name_to_id_dict[dataset_name]) 
    X = dataset.data.features
    y = dataset.data.targets

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Identify numeric and categorical columns
    categorical_columns_selector = selector(dtype_include=object)
    numeric_columns_selector = selector(dtype_exclude=object)
    numeric_features = numeric_columns_selector(X)
    categorical_features = categorical_columns_selector(X)
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OrdinalEncoder())
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit the preprocessor on the training data and transform both train and test sets
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor



# Example usage
if __name__ == "__main__":

    args = parse_arguments()

    X_train, X_test, y_train, y_test, preprocessor = preprocess_dataset(args.dataset_name, args.test_size, args.seed)
    
    print ("Dataset: ", args.dataset_name)
    print ("Test size: ", args.test_size)
    print ("Seed: ", args.seed)
    print ("X_train: ", X_train.shape)
    print ("X_test: ", X_test.shape)
    print ("y_train: ", y_train.shape)
    print ("y_test: ", y_test.shape)

    save_dir = os.path.join(args.output_dir, args.dataset_name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print ("Saving data to: ", save_dir)

    # save the data
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)









