def countwords(a):
    list = a.split()
    dict={}
    count = list.len()
    for i in list:
        x = list.count(i)
        dict[i] = x
    return dict
a = input("enter sentence:")
print(countwords(a))
