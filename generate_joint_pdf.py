import manipulate_data as m
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np
import pickle


def normalize(my_diction):
    total = 0
    for i in my_diction:
        total = total + my_diction[i]
    for j in my_diction:
        my_diction[j] = (float)(my_diction[j])/total
    return my_diction


def add_number_of_lates_column(customers):
    customers["Z6"] = customers["X6"]+customers["X7"]+customers["X8"]+customers["X9"]+ \
                        customers["X10"]+customers["X11"]

    return customers


def add_binned_late_scores_column(customers):
    """
    Add a column for binned late score.
    We just have exclusively different variables now.

    :param customers:
    :return: customers:
    """

    modified_series = customers["Z6"].apply(put_latescore_in_bin)


    customers["bZ6"] = modified_series

    return customers


def put_latescore_in_bin(latescore):
    if (((latescore + 1) % 3) == 0):
        newlatescore = latescore + 1
    elif (((latescore - 1) % 3) == 0):
        newlatescore = latescore - 1
    else:
        newlatescore = latescore

    if newlatescore > 12:
        newlatescore = 12
    elif newlatescore < -12:
        newlatescore = -12


    return newlatescore


def bin_count_dictionary(c):
    newdict = {}



    for key, value in c.items():

        newkey = put_latescore_in_bin(key)

        if newkey in newdict:
            newdict[newkey] += c[key]
        else:
            newdict[newkey] = c[key]

    doomed_keys=[]
    for key, value in newdict.items():
        if key > 12:
            newdict[12] += value
            doomed_keys.append(key)

    for key in doomed_keys:
        del(newdict[key])

    return newdict


def get_stats_on_each_col(customers):
    only_numbers = customers.applymap(np.isreal).all(1)
    customers=customers[only_numbers]
    col_names=[col_name for col_name in customers]
    customers[col_names]=customers[col_names].astype(float)
    stats = customers.describe(include='all').round(2)

    return stats


def get_distribution_over_gender(customers):
    """
    Get the distribution of customers over gender.
    :param customers: PandasDataframe - A table of customers, only customers, the row
    id gives the customer ID
    :return: gender_dist: dictionary - Counts for each gender.
    """
    gender_dist = {}

    males = customers.X2 == 1 # a boolean series indicating male
    females = customers.X2 == 2 # a boolean series indicating female

    num_males = len(customers[males])
    num_females = len(customers[females])

    gender_dist['m'] = num_males
    gender_dist['f'] = num_females

    return normalize(gender_dist)


def get_dist_over_default_status(customers, education_status, gender, latescore, age):
    """
    Get the distribution over default status conditioned on
    gender, education status, and latescore.
    :param customers: PandasDataFrame -> A table of customers
    identified by their customer id.
    :param education_status: integer -> 4 unknown, 3 hs, 2 college, 1 graduate
    :param gender: integer -> 1 male, 2 female
    :param latescore: integer -> -12, -9, ... 0 ... , 3, 6, 9, 12
    :param age: integer -> The age of the customer
    :return: conditional_dist: dictionary ->conditioned probabilities for
    default.
    """
    conditional_dist = {}

    c_edu = customers["X3"] == education_status
    c_gen = customers["X2"] == gender
    c_las = customers["bZ6"] == latescore
    if age > 35:
        c_age = customers["X5"]>35
    else:
        c_age = customers["X5"]<=35

    subset = customers[c_edu & c_gen & c_las & c_age]

    num_def = len(subset[subset["Y"] == 1])
    num_pay = len(subset[subset["Y"] == 0])
    total = num_pay + num_def
    if total == 0:
        conditional_dist['d'] = .5 #baysian assumption
        conditional_dist['p'] = .5
    else:
        conditional_dist['d'] = num_def/total
        conditional_dist['p'] = num_pay/total

    return conditional_dist


def get_ls_dist(customers, education_status, gender, default, age):
    """

    :param customers: Pandas DataFrame -> a table with customers identified by customer id
    :param education_status: int -> 1 - grad, 2 - college, 3 - hs, 4 - other
    :param gender: int -> 1 - male, 2 - female
    :param default: int -> 1 - yes, 2 - no
    :param age: int -> actual age
    :return: ls_dist: dictionary -> distribution over late scores
    """

    ls_dist = {}

    c_edu = customers["X3"]==education_status
    c_gen = customers["X2"]==gender
    c_def = customers["Y"]==default
    if age > 35:
        c_age = customers["X5"]>35
    else:
        c_age = customers["X5"]<=35

    subset = customers[c_edu & c_gen & c_def & c_age]

    late_scores = subset["Z6"]
    counts = Counter(late_scores)
    ls_dist = bin_count_dictionary(counts)
    tot_customers = len(customers)


    # for key, value in ls_dist.items():
    #     ls_dist[key] = value/tot_customers


    return ls_dist


def row_operation(row):
    """
    operate on a row from the test dataframe
    :param row:
    :return: prob for default
    """

    prob_default = look_up(row["X3"], row["X2"], row["bZ6"], row["X5"])

    # prob_default = row["X3"] + row["X2"]

    return prob_default


def look_up(education, gender, latescore, age):
    if age > 35:
        rage = 42
    else:
        rage = 22

    ce = joint_pdf["X3"] == education
    cg = joint_pdf["X2"] == gender
    cl = joint_pdf["bZ6"] == latescore
    ca = joint_pdf["X5"] == rage

    the_row = joint_pdf[ce & cg & cl & ca]
    scuba_gear = the_row.index.values
    prob_default = the_row.at[scuba_gear[0], "default_pred"]

    return prob_default


def build_joint_pdf(customers_train):

    age = [22 ,42]
    gender = [1, 2]
    latescore = [-12, -9, -6, -3, 0, 3, 6, 9, 12]
    education = [1, 2, 3, 4]
    default = 1


    joint_pdf = pd.DataFrame(columns=["X3", "X2", "bZ6", "X5", "default_pred"])

    i = 0
    for a in age:
        for g in gender:
            for l in latescore:
                for e in education:
                    dist = get_dist_over_default_status(customers_train, e, g, l, a)
                    prob_default = dist['d']
                    new_row = [e, g, l, a, prob_default]
                    joint_pdf.loc[i] = new_row
                    i+=1

    return joint_pdf


def produce_error_file():
    starts = list(reversed(list(range(22250, 30000, 50))))
    errors = []
    customers_train = customers[:22250]
    joint_pdf = build_joint_pdf(customers_train)


    for i in starts:

        customers_test = customers[i:]
        customers_test["pred"] = customers_test.apply(row_operation, axis=1)
        customers_test.is_copy = False
        actual = customers_test["Y"].sum()
        predicted = customers_test["pred"].sum()


        #
        error = abs(actual - predicted)/actual
        error = round(error*100, 3)
        print("trial " + str(i) + ": Error-> ", end="")
        print(error)

        errors.append(error)
        pickle.dump(errors, open("errors_list.p", "wb"))


def produce_error_plot():

    with open("errors_list.p", "rb") as f:
        errors = pickle.load(f)

    starts = list(reversed(list(range(22250, 30000, 50))))
    xax = [30000-a for a in starts]



    plt.plot(xax, errors)
    plt.title("Model Error as a Function of Group Size")
    plt.xlabel("Group Size")
    plt.ylabel("Percent Error")
    plt.show()


def produce_joint_pdf_plots(customers, e):
    left_left = get_ls_dist(customers, e, 1, 1, 22)
    left_right = get_ls_dist(customers, e, 1, 1, 42)
    right_left =  get_ls_dist(customers, e, 1, 0, 22)
    right_right = get_ls_dist(customers, e, 1, 0, 42)

    plt.subplot(241)
    plt.bar(left_left.keys(), left_left.values(), 2.7)
    plt.ylim(0,400)
    plt.xticks([])
    plt.title("default \n young \n male")

    plt.subplot(242)
    plt.bar(left_right.keys(), left_right.values(), 2.7)
    plt.yticks([])
    plt.xticks([])
    plt.ylim(0,400)
    plt.title("default \n old \n male")

    plt.subplot(243)
    plt.bar(right_left.keys(), right_left.values(), 2.7)
    plt.yticks([])
    plt.xticks([])
    plt.ylim(0,400)
    plt.title("pay \n young \n male")

    plt.subplot(244)
    plt.bar(right_right.keys(), right_right.values(), 2.7)
    plt.yticks([])
    plt.xticks([])
    plt.ylim(0,400)
    plt.title("pay \n old \n male")

    left_left = get_ls_dist(customers, e, 2, 1, 22)
    left_right = get_ls_dist(customers, e, 2, 1, 42)
    right_left = get_ls_dist(customers, e, 2, 0, 22)
    right_right = get_ls_dist(customers, e, 2, 0, 42)



    plt.subplot(245)
    plt.bar(left_left.keys(), left_left.values(), 2.7)
    plt.ylim(0, 400)
    plt.xlabel("default \n young \n female")

    plt.subplot(246)
    plt.bar(left_right.keys(), left_right.values(), 2.7)
    plt.yticks([])
    plt.ylim(0, 400)
    plt.xlabel("default \n old \n female")

    plt.subplot(247)
    plt.bar(right_left.keys(), right_left.values(), 2.7)
    plt.yticks([])
    plt.xlabel("pay \n young \n female")
    plt.ylim(0, 400)

    plt.subplot(248)
    plt.bar(right_right.keys(), right_right.values(), 2.7)
    plt.yticks([])
    plt.ylim(0, 400)
    plt.xlabel("pay \n old \n female")

    plt.show()


def produce_prob_default_graphs(customers, education):
    """
    Produce the profbabilty for default as a function
    of late score, education, gender, and age.
    :param customers:
    :param education:
    :return:
    """
    left_left = get_ls_dist(customers, e, 1, 1, 22)
    left_right = get_ls_dist(customers, e, 1, 1, 42)
    right_left =  get_ls_dist(customers, e, 1, 0, 22)
    right_right = get_ls_dist(customers, e, 1, 0, 42)

    for key, value in left_left.items():
        other_value=right_left[key]
        total=value+other_value
        left_left[key]=value/total

    plt.subplot(221)
    plt.bar(left_left.keys(), left_left.values(), 2.7)
    plt.title("young \n male")

    for key, value in left_right.items():
        other_value=right_right[key]
        total=value+other_value
        left_right[key]=value/total

    plt.subplot(222)
    plt.bar(left_right.keys(), left_right.values(), 2.7)
    plt.yticks([])
    plt.title("old \n male")




    left_left = get_ls_dist(customers, e, 2, 1, 22)
    left_right = get_ls_dist(customers, e, 2, 1, 42)
    right_left = get_ls_dist(customers, e, 2, 0, 22)
    right_right = get_ls_dist(customers, e, 2, 0, 42)

    for key, value in left_left.items():
        other_value=right_left[key]
        total=value+other_value
        left_left[key]=value/total

    plt.subplot(223)
    plt.bar(left_left.keys(), left_left.values(), 2.7)
    plt.xlabel("young \n female")

    for key, value in left_right.items():
        if right_right.get(key):
            other_value=right_right[key]
            total=value+other_value
            left_right[key]=value/total
        else:
            left_right[key]=1

    plt.subplot(224)
    plt.bar(left_right.keys(), left_right.values(), 2.7)
    plt.yticks([])
    plt.xlabel("old \n female")
    plt.show()


if __name__ == '__main__':


    customers = m.get_all_customers('defaults.xlsx')
    customers = customers[1:]
    customers = add_number_of_lates_column(customers)
    customers = add_binned_late_scores_column(customers)
    customers["X3"] = customers["X3"].replace([-4, -3, -2, -1, 0, 5, 6, 7, 8, 9, 10, 11, 12],4)

    e = 1

    produce_joint_pdf_plots(customers, e)


