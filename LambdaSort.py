
# coding: utf-8

# In[7]:


import operator
my_list =[["John", 1, "a"], ["Larry", 0, "c"], ["Mary", -5, "b"]]
my_list.sort(key = lambda x: x[1])
print(my_list)
my_list.sort(key = operator.itemgetter(2))
print(my_list)

