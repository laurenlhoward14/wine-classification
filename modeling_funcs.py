# General libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Connect with AWS
from sqlalchemy import create_engine

# Modeling functions
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.dummy import DummyClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import tree

def print_confusion_matrix(confusion_matrix, class_names=['White', 'Red'], figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label');
    return plt.figure(figsize=figsize);


def roc_auc(model, name, y_val, X_val):
    '''Plots ROC Curve and AUC Score
    Inputs: Model to plot for, Name of Model, x and y test values
    '''
    fpr, tpr, thresholds = roc_curve(y_val, model.predict_proba(X_val)[:,1], pos_label=1)
    plt.plot(fpr, tpr,lw=2)
    plt.plot([0,1],[0,1],c='violet',ls='--')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])


    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC curve for {name}');
    print("ROC AUC score = ", roc_auc_score(y_val, model.predict_proba(X_val)[:,1]))


def incorrect_rows(y_val, y_predictions):
  '''Returns a list of indexes where the predictions are incorrect
  Args:
  y_val - Actual y values
  y_predictions - y predictions from model

  Output Type: List
  '''

  # Idx:Actuals
  y_values = dict(y_val)
  y_predicted = list(y_predictions)

  # Split dict into list of idx and actuals
  indexes = []
  actuals = []
  for key, value in y_values.items():
      indexes.append(key)
      actuals.append(value)

  # Create a list of tuples (actual values, predictions)
  acts_preds = list(zip(actuals, y_predicted))

  # Iterate through and determine whether prediction was correct or not
  predictions = []
  for idx,(act,pred) in enumerate(acts_preds):
      if act == pred:
          predictions.append("Correct")
      else:
          predictions.append("Incorrect")

  # Create dictionary of indexes and predictions
  total = dict(zip(indexes, predictions))

  # Iterate through dictionary - if "Incorrect" in value append keys and values and create a new dictionary
  incorrect_preds_idx = []
  incorrect_preds_values = []
  for key, value in total.items():
      if value == 'Incorrect':
          incorrect_preds_idx.append(key)

  return incorrect_preds_idx


def incorrect_df(y_val, y_predictions, probability_array, dataframe):
    """Returns a dataframe of incorrectly predicted samples using all
    features from original dataframe. Shows additional columns of the models
    predicted probabilities for each feature.

    Args:
    y_val - actual y values
    y_predictions - y_predictions from model
    probability_array - probabilities calculated by model, found by calling .predict_proba on model
    dataframe - original dataframe with all data used for model. Defaults to wine.

    Output Type: Dataframe
    """

    # Idx:Actuals
    y_values = dict(y_val)
    y_predicted = list(y_predictions)

    # Split dict into list of idx and actuals
    indexes = []
    actuals = []
    for key, value in y_values.items():
        indexes.append(key)
        actuals.append(value)

    # Create a list of tuples (actual values, predictions)
    acts_preds = list(zip(actuals, y_predicted))

    # Iterate through and determine whether prediction was correct or not
    predictions = []
    for idx,(act,pred) in enumerate(acts_preds):
        if act == pred:
            predictions.append("Correct")
        else:
            predictions.append("Incorrect")


    # Create a list of probabilites for each training value
    proba_list = np.ndarray.tolist(probability_array)

    white_prob = []
    red_prob = []
    for item in proba_list:
        white_prob.append(round(item[0], 4))
        red_prob.append(round(item[1], 4))

    # Create list of lists - Predictions/White Prob/Red Prob
    pred_probs = list(zip(predictions, white_prob, red_prob))

    # Create dictionary of indexes and predictions,probabilities
    total = dict(zip(indexes, pred_probs))

    # Iterate through dictionary - if "Incorrect" in value append keys and values and create a new dictionary
    incorrect_preds_idx = []
    incorrect_preds_values = []
    for key, value in total.items():
        if 'Incorrect' in value:
            incorrect_preds_idx.append(key)
            incorrect_preds_values.append(value)
    incorrect_total = dict(zip(incorrect_preds_idx, incorrect_preds_values))

    # Create dataframe of incorrect predictions and probabilities
    probs_df = pd.DataFrame.from_dict(incorrect_total, orient='index', columns=["incorrect", "white_prob", "red_prob"])
    probs_df.drop("incorrect", axis=1, inplace=True)

    # Locate rows in which the model incorrectly predicted from initial database
    incorrect_preds = dataframe.iloc[incorrect_preds_idx]

    # Merge two dataframes
    return pd.merge(incorrect_preds, probs_df, left_index=True, right_index=True)

def log_reg_errors(y_val, prediction, x_train, y_train):

    '''Returns Accuracy, Precision, Recall and F1 Scores for Logistic
    Regression and Cross Validation models.

    Inputs: Y validation data
            Logistic Regression Predictions
            X training data
            Y training data'''

    cross_val_model = LogisticRegressionCV(random_state=14).fit(x_train, y_train)

    accuracy = (accuracy_score(y_val, prediction)).round(3)
    cross_val_acc = (cross_val_score(cross_val_model, x_train, y_train)).mean().round(3)
    precision = (precision_score(y_val, prediction)).round(3)
    cross_val_prec = (cross_val_score(cross_val_model, x_train, y_train, scoring='precision')).mean().round(3)
    recall = (recall_score(y_val, prediction)).round(3)
    cross_val_recall = (cross_val_score(cross_val_model, x_train, y_train, scoring='recall')).mean().round(3)
    f1 = (f1_score(y_val, prediction)).round(3)
    cross_val_f1 = (cross_val_score(cross_val_model, x_train, y_train, scoring='f1')).mean().round(3)

    return(f"""Accuracy: {accuracy} CV:{cross_val_acc}
    Precision: {precision} CV:{cross_val_prec}
    Recall: {recall} CV:{cross_val_recall}
    F1: {f1} CV: {cross_val_f1}""")


def append_to_lists(validation, prediction, predictionprobability, name):
    """Appends correct details to the lists that were created to keep track of each model performance
    validation = validation data
    prediction = prediction values
    predictionprobability = probability of predictions
    name = (string) Name of the model and which features used to distinguish between models"""

    incorrect_list = incorrect_rows(validation, prediction)
    incorrect_dataframe = incorrect_df(validation, prediction, predictionprobability, wine)

    model_name.append(name)
    incorrect_predictions.append(len(incorrect_list))
    list_of_incorrects.append(incorrect_list)
