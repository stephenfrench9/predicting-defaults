import manipulate_data as m
import matplotlib.pyplot as plt
from collections import Counter
import math
import pandas as pd

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

    print("DO WE HAVE ACCESS TO JOINT PDF")

    print(joint_pdf.head())

    the_row = joint_pdf[ce & cg & cl & ca]

    print("FROM LOOK UP FUNCTION: LENGTH OF the row: ", end="" )
    print(len(the_row))

    print("We are trying to match: " + "educa5ti: " + str(education) + "     genader: " + str(gender)
          + "   latescore: " + str(latescore) + "   age: " + str(rage) )

    print("The row type: ", end="")
    print(type(the_row))

    print("The row")
    print(the_row)

    print("The row labels")
    print(the_row.index.values)

    scuba_gear = the_row.index.values

    print("The int")
    print(type(scuba_gear[0]))

    print("the int, actualy")
    print(scuba_gear[0])

    prob_default = the_row.at[scuba_gear[0], "default_pred"]
    print("FROM LOOK UP FUNCTION: The prob is: " + str(prob_default))
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


if __name__ == '__main__':
    customers = m.get_all_customers('defaults.xlsx')
    customers = customers[1:]
    customers = add_number_of_lates_column(customers)
    customers = add_binned_late_scores_column(customers)
    customers["X3"] = customers["X3"].replace([-4, -3, -2, -1, 0, 5, 6, 7, 8, 9, 10, 11, 12],4)

    customers_train = customers[:22250]
    customers_test = customers[22250:]
    customers_test.is_copy = False

    joint_pdf = build_joint_pdf(customers_train)
    customers_test["pred"] = customers_test.apply(row_operation, axis=1)

    print(joint_pdf.head())

    print(customers_test.head())
    customers_test.to_csv('coastal city.csv')


    # accuracy = compare(customers_test)
    # print(accuracy)


  ######################
    #
    # left_left = get_ls_dist(customers, 1, 1, 1, 22)
    # left_right = get_ls_dist(customers, 1, 1, 1, 42)
    # right_left =  get_ls_dist(customers, 1, 1, 0, 22)
    # right_right = get_ls_dist(customers, 1, 1, 0, 42)
    #
    # plt.subplot(211)
    # plt.subplot(141)
    # plt.bar(left_left.keys(), left_left.values(), 2.7)
    # plt.ylim(0,400)
    #
    # plt.subplot(142)
    # plt.bar(left_right.keys(), left_right.values(), 2.7)
    # plt.yticks([])
    # plt.ylim(0,400)
    #
    # plt.subplot(143)
    # plt.bar(right_left.keys(), right_left.values(), 2.7)
    # plt.yticks([])
    # plt.ylim(0,400)
    #
    # plt.subplot(144)
    # plt.bar(right_right.keys(), right_right.values(), 2.7)
    # plt.yticks([])
    # plt.ylim(0,400)
    #
    # plt.subplot(212)
    # plt.plot([1,2,3])
    # plt.show()

#########################



    # defaulters = customers["Y"] == 1
    # payers = customers["Y"] == 0
    #
    # plt.title("All Customers")
    # plt.bar(["payer","defaulter"], [len(customers[payers])/len(customers), len(customers[defaulters])/len(customers)], .8)
    # plt.xlabel("p(default): " + str(round(len(customers[defaulters])/len(customers), 2)))
    # plt.show()
    #
    # man_defaulter = customers[defaulters]["X2"] == 1
    # woman_defaulter = customers[defaulters]["X2"] == 2
    #
    # col = customers[defaulters]["Z6"][1:]
    # c = dict(Counter(col))
    # c = bin_count_dictionary(c)
    # c = normalize(c)
    # plt.figure(1)
    # plt.ylim(0,.5)
    # plt.xlim(-14,14)
    # plt.title("defaults")
    # plt.bar(c.keys(), c.values(), 2.8)
    #
    # col = customers[payers]["Z6"][1:]
    # c = Counter(col)
    # c = bin_count_dictionary(c)
    # c = normalize(c)
    # plt.figure(2)
    # plt.ylim(0,.5)
    # plt.xlim(-14,14)
    # plt.title("payers")
    # plt.bar(c.keys(), c.values(), 2.8)
    # plt.show()
    #
    # col = customers[defaulters][man_defaulter]["Z6"][1:]
    # c = Counter(col)
    # c = bin_count_dictionary(c)
    # c = normalize(c)
    # plt.figure(1)
    # plt.ylim(0,.5)
    # plt.xlim(-14,14)
    # plt.title("man defaulters")
    # plt.bar(c.keys(), c.values(), 2.8)
    #
    # col = customers[defaulters][woman_defaulter]["Z6"][1:]
    # c = Counter(col)
    # c = bin_count_dictionary(c)
    # c = normalize(c)
    # plt.figure(2)
    # plt.ylim(0,.5)
    # plt.xlim(-14,14)
    # plt.title("woman defaulters")
    # plt.bar(c.keys(), c.values(), 2.8)
    # plt.show()
