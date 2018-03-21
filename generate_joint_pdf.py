import manipulate_data as m
import matplotlib.pyplot as plt
from collections import Counter

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

if __name__ == '__main__':
    customers = m.get_all_customers('defaults.xlsx')
    customers = customers[1:]

    customers = add_number_of_lates_column(customers)
    defaults = customers["Y"] == 1
    payers = customers["Y"] == 0


    col = customers[defaults]["Z6"][1:30000]
    c = Counter(col)
    c = normalize(c)
    plt.figure(1)
    plt.title("defaults, no cleaning")
    plt.bar(c.keys(), c.values())

    col = customers[payers]["Z6"][1:30000]
    c = Counter(col)
    c = normalize(c)
    plt.figure(2)
    plt.title("payers, no cleaning")
    plt.bar(c.keys(), c.values())
    plt.show()
