import streamlit as st
import pandas as pd
import numpy as np
import gzip
import dill

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from xgboost import XGBClassifier

from sklearn.calibration import CalibratedClassifierCV

import warnings
warnings.filterwarnings('ignore')
st.title('Fraud Detection')

dataset_name = st.sidebar.selectbox('Select dataset', ("Credit Card Fraud Detection", 'upload'))
st.write(dataset_name)
@st.cache(suppress_st_warning=True)
def get_dataset(dataset_name):
    if dataset_name == 'Credit Card Fraud Detection':
        data = pd.read_csv('https://raw.github.com/HamoyeHQ/g01-fraud-detection/master/data/credit_card_dataset.zip')

        df = data.copy()
        y = df.pop('Class')
        X = df
        return X, y

cols = ['V' + str(i) for i in range(1, 29) if i != 25]
X, y = get_dataset(dataset_name)
st.write('shape of dataset', X.shape)
st.write('first five rows of dataset', X.head())

admin_cost = 2.5

@st.cache(suppress_st_warning=True)
# defining a function to calculate cost savings
def cost_saving(ytrue, ypred, amount, threshold=0.5, epsilon=1e-7):
    ypred = ypred.flatten()
    fp = np.sum((ytrue == 0) & (ypred == 1))
    cost = np.sum(fp * admin_cost) + np.sum((amount[(ytrue == 1) & (ypred == 0)]))
    max_cost = np.sum((amount[(ytrue == 1)]))
    savings = 1 - (cost / (max_cost + epsilon))

    return savings

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cols=cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return np.array(X[self.cols])

        elif isinstance(X, pd.Series):
            return np.array(X[self.cols]).reshape(1, -1)

        elif isinstance(X, np.ndarray):
            self.cols_ind = [int(col[1:]) for col in self.cols]
            if len(X.shape) == 1: # if one dimensional array
                return X[self.cols_ind].reshape(1, -1)
            return X[:, self.cols_ind]

        else:
            raise TypeError('expected input type to be any of pd.Series, pd.DataFrame or np.ndarray but got {}'.format(type(X)))

cols_select = ColumnSelector()
scaler = StandardScaler()

data_prep = Pipeline([('columns', cols_select), ('scaler', scaler)]) # data preparation pipeline

X_prep = data_prep.fit_transform(X, y)  # fitting and transforming the data

model = XGBClassifier(random_state=1)

sample_weights = np.array([X['Amount'].iloc[ind] if fraud else admin_cost for ind, fraud in enumerate(y.values)])

#%%time

if st.button('Predict'):

    model.fit(X_prep, y, sample_weight=sample_weights);


    # defining function to get predictions
    def get_predictions(X, proba=False):
        # loading in useful objects
        with gzip.open('data_prep_pipe.gz.dill', 'rb') as f:
            data_prep = dill.load(f)

        # model = XGBClassifier(random_state=1)

        # model.load_model('xgboost.bin')

        with gzip.open('calibration.gz.dill', 'rb') as f:
            calibration = dill.load(f)

        Xt = data_prep.transform(X)  # prepare (preprocess) the user's input

        if proba:
            pred = calibration.predict_proba(Xt)  # gets the probability of belonging to the positvie class

            if len(pred.shape) > 1:  # pred is 2-dim (multi-input)
                pred = pred[:, 1]

            else:  # pred is 1-dim (single-input)
                pred = pred[1]

        else:  # get raw predictions
            pred = calibration.predict(Xt)  # gets the prediction

        return pred


    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)


    def prediction_summary(user_input, ytrue=None):
        """
          This function is both for testing our model when we know the true label of user's input and getting only predictions when we don't know the true labels.

          Args:
            user_input:
              type: #any of numpy array, pandas Series or dataframe.

              User's input is expected to be for all features apart from 'Class' feature making them 30 in number as arranged in our dataset.

            y_true:
              type: #any of numpy array or pandas Series.
              The true labels for user_input

        Return:
            a dataframe of 'Class' and the probability of 'Class' being fraud. A 'Class' of 1 means fraud, while 0 means not fraud. If ytrue is given;
            f1_score and cost saving are also printed out.
        """

        proba = get_predictions(user_input, proba=True)
        pred = get_predictions(user_input)
        pred_df = pd.DataFrame({'Class': pred, 'Fraud_Probabilty': proba})

        if ytrue is not None:  # if we know the true labels, it means we want to test the model and printing out metrics will be useful

            if len(user_input.shape) > 1:  # if the input has more than 1 row (multi-input)
                f1 = ('f1_score is {}'.format(f1_score(ytrue, pred)))
                if isinstance(user_input, np.ndarray):
                    amount = user_input[:, -1]
                else:
                    amount = user_input.iloc[:, -1]
                cs = ('cost saving is {}'.format(cost_saving(ytrue, pred, amount)))

            else:  # a single input.
                f1 = ('f1_score is {}'.format(f1_score(ytrue, pred)))
                cs = ('cost saving is {}'.format(cost_saving(ytrue, pred, user_input[-1].reshape(1))))

        return pred_df, f1, cs  # in any case, finally return the dataframe of predictions.


    result, f1, cs = prediction_summary(X_test, y_test)  # 71,202 inputs
    st.write(result.head())
    st.write(f1)
    st.write(cs)