
# coding: utf-8

# In[13]:


def x_raised_n(x, n):
    try:
        if n == 0:
            raise Exception
    except Exception:
        print("0 cannot be entered to function.")
        return
    
    if (n > 0):
        if (n == 1):
            return(x)
        else:
            return(x * x_raised_n(x, n-1))
    else:
        if (abs(n) == 1):
            return(1/x)
        else:
            return(1/x * x_raised_n(1/x, abs(n)-1))
                 
print(x_raised_n(3, 3))
print(x_raised_n(3, -3))
print(x_raised_n(3, 0))
    

