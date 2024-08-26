from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ucimlrepo import fetch_ucirepo

def preprocess_UCI_dataset(dataset_id, encoding_type, test_size=0.2, random_state=42):

    # Load the dataset
    dataset = fetch_ucirepo(id=dataset_id)
    print ("Fetched dataset name:", dataset.metadata.name)
    X = dataset.data.features
    y = dataset.data.targets

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Identify numeric and categorical columns
    categorical_columns_selector = selector(dtype_include=object)
    numeric_columns_selector = selector(dtype_exclude=object)
    numeric_features = numeric_columns_selector(X)
    categorical_features = categorical_columns_selector(X)

    print ("Numeric features: ", numeric_features)
    print ("Categorical features: ", categorical_features)
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    Encoder = OneHotEncoder() if encoding_type == "onehot" else OrdinalEncoder()
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', Encoder)
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









