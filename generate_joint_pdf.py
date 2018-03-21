import manipulate_data as m
import matplotlib.pyplot as plt
from collections import Counter
import math

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


def bin_count_dictionary(c):
    newdict = {}



    for key, value in c.items():
        if(((key+1)%3)==0):
            newkey = key + 1
        elif(((key-1)%3)==0):
            newkey = key - 1
        else:
            newkey = key

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

if __name__ == '__main__':
    customers = m.get_all_customers('defaults.xlsx')
    customers = customers[1:]

    customers = add_number_of_lates_column(customers)
    defaulters = customers["Y"] == 1
    payers = customers["Y"] == 0

    plt.title("All Customers")
    plt.bar(["payer","defaulter"], [len(customers[payers])/len(customers), len(customers[defaulters])/len(customers)], .8)
    plt.xlabel("p(default): " + str(round(len(customers[defaulters])/len(customers), 2)))
    plt.show()

    man_defaulter = customers[defaulters]["X2"] == 1
    woman_defaulter = customers[defaulters]["X2"] == 2

    col = customers[defaulters]["Z6"][1:]
    c = dict(Counter(col))
    c = bin_count_dictionary(c)
    c = normalize(c)
    plt.figure(1)
    plt.ylim(0,.5)
    plt.xlim(-14,14)
    plt.title("defaults")
    plt.bar(c.keys(), c.values(), 2.8)

    col = customers[payers]["Z6"][1:]
    c = Counter(col)
    c = bin_count_dictionary(c)
    c = normalize(c)
    plt.figure(2)
    plt.ylim(0,.5)
    plt.xlim(-14,14)
    plt.title("payers")
    plt.bar(c.keys(), c.values(), 2.8)
    plt.show()

    col = customers[defaulters][man_defaulter]["Z6"][1:]
    c = Counter(col)
    c = bin_count_dictionary(c)
    c = normalize(c)
    plt.figure(1)
    plt.ylim(0,.5)
    plt.xlim(-14,14)
    plt.title("man defaulters")
    plt.bar(c.keys(), c.values(), 2.8)

    col = customers[defaulters][woman_defaulter]["Z6"][1:]
    c = Counter(col)
    c = bin_count_dictionary(c)
    c = normalize(c)
    plt.figure(2)
    plt.ylim(0,.5)
    plt.xlim(-14,14)
    plt.title("woman defaulters")
    plt.bar(c.keys(), c.values(), 2.8)
    plt.show()
