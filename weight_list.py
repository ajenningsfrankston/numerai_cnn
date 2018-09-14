from itertools import product

#
# generates the list of weight vectors
# for searching
#


def weight_list():
    a = list(product([1, 2, 3, 4], repeat=2))
    b = []
    for x in a:
        b.append(list(x))
    c = []
    for x in b:
        c.append([y * 0.25 for y in x])
    d = []
    for x in c:
        x.append(1.0)
        d.append(x)
    return c

