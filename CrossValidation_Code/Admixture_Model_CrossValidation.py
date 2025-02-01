import numpy as np
from scipy.optimize import minimize
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
#%%
# Load Data
path = "df_new_results.csv"
df = pd.read_csv(path)

#%%
def calc_NN(df):
    M = len(df.columns) - 1
    feature_results = []
    unique_letters_final = []
    
    for col in df.columns[:-1]:  
        unique_letters = sorted(set(''.join(df[col])))
        letter_to_number = {letter: i for i, letter in enumerate(unique_letters)}
        unique_letters_final.append(unique_letters)
        col_results = []
        for row in df[col]:
            row_occurrences = [0] * len(unique_letters)
            for letter in row:
                row_occurrences[letter_to_number[letter]] += 1
            col_results.append(row_occurrences)
        feature_results.append(col_results)
    positions_of_N = []
    for m in range(M):
        N_position = None
        for idx in range(len(unique_letters_final[m])):
            if unique_letters_final[m][idx] == 'N':
                N_position = idx
                break
        positions_of_N.append(N_position)
    result_dict =  {df.columns[0]: df[df.columns[0]].tolist()}
    for i, col_name in enumerate(df.columns[:-1]):
        result_dict[col_name] = feature_results[i]
    df_res = pd.DataFrame(result_dict)
    df_res["Population"] = df.iloc[:, 104]
    return df_res, positions_of_N

def sum_lists(lists):
    return list(np.sum(lists, axis=0))

def normalize_features(df, group_sizes):
    for group in group_sizes.index:
        group_size = group_sizes[group]
        group_indices = df[df['Population'] == group].index
        for col in df.columns[1:]:
            df.loc[group_indices, col] = df.loc[group_indices, col].apply(lambda lst: [x / (2*group_size) for x in lst])
    return df


# Determine Allele Frequencies
def calc_p(df_in, ind=None):
    
    summed_df = df_in.groupby('Population').agg(
        lambda x: sum_lists(x.tolist()) if isinstance(x.iloc[0], list) else x.iloc[0]
    ).reset_index()
    group_sizes = df_in['Population'].value_counts()
    normalized_df = normalize_features(summed_df, group_sizes)
    df = normalized_df
    features = df.columns[1:]  
    groups = df['Population'].unique()  
    p_values = [[] for _ in features]

    for i, feature in enumerate(features):
        for group in groups:
            group_values = df[df['Population'] == group][feature].tolist()
            p_values[i].append(group_values[0])

    return p_values

# Determine datax in the correct format 
def save_x(df): 
    rows_list = []
    for index, row in df.iterrows():
        row_dict = row.to_dict()
        row_list = [row_dict[col] for col in df.columns[1:]]
        rows_list.append(row_list)
    return(rows_list)


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

# Estimator for the IAs
def l(q,x, p):
    M = len(x)
    res1 = 0
    for m in range(M):
        J = len(x[m])
        for j in range(J):
            result = p[m]
            p_tran = transpose(result)
            loc = np.dot(q, p_tran[j])
            x_temp = x[m][j]
            if(loc > 0 and loc < 1):
                res1 += x_temp * np.log(loc) #+ (2-x_temp) * np.log(1-loc)
    return -res1

def constraint(q):
    return np.sum(q) - 1 


# Returns the prediction probabilities
def rep_q(x_test, K, X_train):
    p = calc_p(X_train)
    x_test = save_x(x_test)
    q_all = []
    N = len(x_test)
    b = [(0, 1) for _ in range(K)]  
    for i in range(N):
        x = x_test[i]
        q0 = np.random.rand(K)
        q0 /= np.sum(q0)  # 
        cons = ({'type': 'eq', 'fun': constraint})
        result = minimize(l, q0, args=(x, p), constraints=cons, bounds = b)
        hat_q = result.x
        q_all.append(hat_q)
    return q_all


class CustomClassifier(BaseEstimator):
    def __init__(self, K=8, random_state=None):
        """
        Initialize the classifier.
        K: Number of mixture groups for admixture proportions.
        random_state: Seed for random number generation.
        """
        self.K = K
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the model to the data.
        X: Features (input data)
        y: Target labels
        """
        self.X_train = X
        self.y_train = y
        self.df_in = X  # Store input data for later use
        self.p = calc_p(self.df_in)  # Calculate p-values (allele frequencies)
        self.x_values = save_x(self.df_in)  # Process the input data
        return self

    def predict(self, X):
        """
        Predict the labels for new data.
        X: Features of the new data.
        Returns: Predicted group labels.
        """
        x_test = save_x(X)  # Process test data
        q_hat = self.rep_q(x_test, self.K, self.X_train)  # Calculate the q-values for test data
        return np.argmax(q_hat, axis=1)  # Return the group with the highest probability

    def predict_proba(self, X):
        """
        Predict the probabilities of each class for the test data.
        X: Features of the new data.
        Returns: Probabilities of each group for the input data.
        """
        x_test = save_x(X)  # Process test data
        q_hat = self.rep_q(x_test, self.K, self.X_train)  # Calculate the q-values for test data
        return q_hat  # Return the probabilities (admixture proportions)

    def rep_q(self, x_test, K, X_train):
        """
        Replicate the q-value calculation for multiple test samples.
        x_test: Test data.
        K: Number of groups.
        X_train: Training data (used to calculate allele frequencies).
        Returns: List of admixture proportions for each test sample.
        """
        p = calc_p(X_train)  # Calculate p-values (allele frequencies) from the training data
        q_all = []
        N = len(x_test)
        b = [(0, 1) for _ in range(K)]  # Bound for admixture proportions (between 0 and 1)

        for i in range(N):
            x = x_test[i]
            q0 = np.random.rand(K)
            q0 /= np.sum(q0)  # Normalize to sum to 1
            cons = ({'type': 'eq', 'fun': constraint})  # Ensure the sum of admixture proportions is 1
            result = minimize(l, q0, args=(x, p), constraints=cons, bounds=b)  # Minimize the loss function
            hat_q = result.x  # Predicted admixture proportions
            q_all.append(hat_q)

        return q_all
# K: number of classes
clf = CustomClassifier(K=9)

#%%

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)# Listen zum Speichern der Ergebnisse
accuracies = []
roc_aucs = []
log_losses = []
true_classes = []
predicted_classes = []
# Labels
y_all = df['Population']
X_temp = calc_NN(df)[0].iloc[:,0:104]

X_temp.insert(0, 'Population', y_all.tolist())

for train_index, test_index in cv.split(X_temp, y_all):
    
    X_train_cv, X_test_cv = X_temp.iloc[train_index], X_temp.iloc[test_index]
    y_train_cv, y_test_cv = y_all.iloc[train_index], y_all.iloc[test_index]

    
    clf.fit(X_train_cv, y_train_cv)
    
    predictions = clf.predict(X_test_cv)
    
    accuracy = accuracy_score(y_test_cv, predictions)
    accuracies.append(accuracy)
    
    proba = clf.predict_proba(X_test_cv)
    proba = [prob / prob.sum() for prob in proba]
    roc_auc = roc_auc_score(y_test_cv, proba, average='macro', multi_class='ovr')
    roc_aucs.append(roc_auc)
    
    log_loss_value = log_loss(y_test_cv, proba)
    log_losses.append(log_loss_value)
    
    true_classes.extend(y_test_cv.tolist())
    predicted_classes.extend(predictions.tolist())
