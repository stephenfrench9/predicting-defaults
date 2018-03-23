import numpy as np
import pandas as pd
import manipulate_data as m
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import scipy
import matplotlib.pyplot as plt

def get_payments(customers):
    """"get all the payments that each customer has made, with
    customers still identified by their ID from the original dataset.

    Return: A table with the payments for each customer.
    The datatype is a Pandas DataFrame.
    """
    all_col = [a for a in customers]
    payments_col = all_col[17:23]
    payments = customers[payments_col]

    return payments


def get_statements(customers):
    """Get all the statements issued to customers. The customers
    are still identified by their customerID, which is in another column
    of the table

    Returns: A table with all the statements issued to each customer.
    The datatype is a Pandas DataFrame.
    """

    all_col = [a for a in customers]

    statements_col = all_col[11:17]
    statements = customers[statements_col]

    return statements


def get_payments_and_statements(customers):
    """Get all the payments and statements for each customers.
    Returns: A table with all the payments and statements for each customer.
    The datatype is a Pandas DataFrame."""

    all_col = [a for a in customers]
    features_col = all_col[11:23]
    features = customers[features_col].astype(float).divide(100).round()
    return features


def get_target_variable(customers):
    """

    :param customers:
    :return: target_np: A numpy array with the default status for each customer.
    """
    all_col = [a for a in customers]
    target_col = all_col[23:24]
    target = customers[target_col].astype(float)
    target_np = target.as_matrix(["Y"]).reshape((target.shape[0],))
    return target_np


if __name__ == '__main__':
    customers = m.get_all_customers('defaults.xlsx')
    customers = customers[1:]

    payments = get_payments(customers)
    statements = get_statements(customers)
    pay_and_state = get_payments_and_statements(customers)
    target_np = get_target_variable(customers)


    pred_data = pay_and_state[["X23"]]

    print(pred_data.head())
    print("A check on pred: ")
    print("pred type: " + str(type(pred_data)))
    print("pred length: " + str(len(pred_data)))


    print(pd.DataFrame(target_np).head())
    print("A check on target: ")
    print("pred type: " + str(type(pred_data)))
    print("pred length: " + str(len(pred_data)))


    k=range(1,20)
    k1=[1,3,5,7,9,11,13,15,17,19]

    acc = []

    for i in k:
        knn = KNeighborsClassifier(i)
        results = cross_val_score(knn, pred_data, target_np, cv=5)
        acc.append(np.average(np.array(results)))


    plt.plot(k,acc)
    plt.xticks(k1,k1)
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.title("Using Payment and Statement Data to Predict Default")
    plt.show()


