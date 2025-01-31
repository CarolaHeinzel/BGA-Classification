import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

def train_and_evaluate_pls_da(X_train, X_test, y_train, y_test, n_components=100):
    # Convert to DataFrame if necessary
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    for col in X_train.columns:
        if X_train[col].dtype == 'object': 
            encoder = LabelEncoder()
            
            combined_data = pd.concat([X_train[col], X_test[col]], axis=0)
            encoder.fit(combined_data)
        
            X_train[col] = encoder.transform(X_train[col])
            X_test[col] = encoder.transform(X_test[col])

    # Binarize the target variable for multi-class classification
    lb = LabelBinarizer()
    Y_train_binarized = lb.fit_transform(y_train)
    Y_test_binarized = lb.transform(y_test)
    lb = LabelBinarizer()
    Y_train_binarized = lb.fit_transform(y_train)
    

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PLS-DA model training
    pls_da = PLSRegression(n_components=n_components)
    pls_da.fit(X_train_scaled, Y_train_binarized)

    # Predictions
    Y_pred_continuous = pls_da.predict(X_test_scaled)
    Y_pred_classes = np.argmax(Y_pred_continuous, axis=1)

    # Encode y_test if necessary
    if isinstance(y_test[0], str):
        y_test_encoded = lb.transform(y_test).argmax(axis=1)
    else:
        y_test_encoded = y_test

    # Metrics calculation
    roc_auc = roc_auc_score(Y_test_binarized, Y_pred_continuous, multi_class='ovr')
    log_loss_value = log_loss(Y_test_binarized, Y_pred_continuous)
    accuracy = accuracy_score(y_test_encoded, Y_pred_classes)
    return roc_auc, log_loss_value, accuracy, Y_pred_continuous, Y_test_binarized


