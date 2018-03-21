"""Manipulate data from the 'Varsity Tutors defaults' dataset. The idea is that
you identify a customer by their customer ID. Each customer has a predictor data like
age, gender, education, and payment history. And each customer has an outcome - they did
or did not default.

    -get all the raw data for a customer identified by their customer ID.
    -get the 'default' outcome for a customer identified by their customer ID
    -display plots/time series of raw data for a given customer.
    -display delinquency curve for a given customer.
    -get all the customers in a pandas dataframe.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_all_customers(filename):
    """Get a table of all customers information. Each row in
    the table is a customer.

    Returns: A pandas data frame """
    raw_excel = pd.ExcelFile(filename)
    customers = raw_excel.parse('credit_defaults.csv')

    return customers


def place_into_list_and_reverse(a_series):
    """Change the datatype for the given object, and
    reverse its order."""
    a_list = []
    length=len(a_series)

    print(type(a_series))

    for i in range(0, length):
        a_list.append(a_series[i])

    return list(reversed(a_list))


def graph_raw_data(customerID, customers):
    """Produce plots of the all the raw data for a give customer.
    View their payment history, their statement history, and
    a record of whether their payment was late or not."""

    months = ['april', 'may', 'june', 'july', 'aug', 'sept']
    one_customer = customers.loc[["ID", customerID],:]

    for i in range(0,24):
        print("{:10}: {}".format(one_customer.iloc[0][i], one_customer.iloc[1][i]))


    numbers = place_into_list_and_reverse(one_customer.iloc[1][5:11])
    labels = place_into_list_and_reverse(one_customer.iloc[0][5:11])
    plt.figure(2)
    plt.plot(months,numbers)
    plt.title("payment status")


    numbers = place_into_list_and_reverse(one_customer.iloc[1][11:17])
    labels = place_into_list_and_reverse(one_customer.iloc[0][11:17])
    plt.figure(3)
    plt.plot(months, numbers)
    plt.title("statement during this period")


    numbers = place_into_list_and_reverse(one_customer.iloc[1][17:23])
    labels = place_into_list_and_reverse(one_customer.iloc[0][17:23])
    plt.figure(4)
    plt.plot(months, numbers)
    plt.title("actual payment")
    plt.show()


def graph_delinquency_curve(customerID, customers):
    """Produce and display the deliquency curve for a certain customer.
    The deliquency curve is a timeseries of portion of the statment that
    they paid.
    """

    months = ['april', 'may', 'june', 'july', 'aug', 'sept']
    one_customer = customers.loc[["ID", customerID],:]

    print("The .loc function produces an object of type: " +str(type(one_customer)))

    payments=np.array(place_into_list_and_reverse(one_customer.iloc[1][17:23]))
    statements=np.array(place_into_list_and_reverse(one_customer.iloc[1][11:17]))
    due_this_month=np.insert(statements, 0, 0)[:-1]

    delinquent = (due_this_month - payments)/due_this_month
    delinquent[0]=0

    is_delinquent = one_customer.iloc[1][23]
    delinq_status = place_into_list_and_reverse(one_customer.iloc[1][5:11])
    delinq_status=[str(a) for a in delinq_status]

    plt.figure(1)
    plt.title("Deliquent? " + str(is_delinquent))
    plt.plot(months, delinquent)
    plt.xticks(months, delinq_status)

    plt.show()


def get_stats_on_cols(customers):
    """Get statistics about the distribution of numbers in
    each column of the 'customers' dataframe.
    Returns: A data frame with stats about the input 'customers'"""
    only_numbers = customers.applymap(np.isreal).all(1)
    customers=customers[only_numbers]

    col_names=[col_name for col_name in customers]

    customers[col_names]=customers[col_names].astype(float)

    stats = customers.describe(include='all').round(2)

    return stats


def get_more_stats_on_cols(customers):
    """Get additional statistics about the 'customers' dataframe.
    These statistics are useful for designing 'binning' strategies
    for the data. Included are statistics like 'number of unique entries'.
    Returns: a dataframe with additional statistics about customers"""
    only_numbers = customers.applymap(np.isreal).all(1)
    customers=customers[only_numbers]

    return customers.describe()

if __name__ == '__main__':
    file_name = 'defaults.xlsx'
    customers = get_all_customers(file_name)

    stats2 = get_more_stats_on_cols(customers)

    print(stats2)







    # print(len(customers))
    # print(only_numbers[:5])
    # print(len(customers))




    # print(customers.info())
    # print(customers.head())

    # graph_raw_data(1, customers)
    # graph_delinquency_curve(1, customers)


    # for i in range(1,200):
    #     print(i)
    #     graph_raw_data(i, customers)
    #     graph_delinquency_curve(i, customers)
    #     plt.show()