
# coding: utf-8

# In[4]:


print(ord('A'))
print(ord('G'))
print(ord('C'))

def is_prime(number):
    for i in range(2, int((number / 2) + 1)):
        if(number % i == 0):
            return False
    return True 

def find_nearest_prime(given_number):
    for i in range(1,int(given_number / 2)):
        if is_prime(given_number - i):
            ch = chr(given_number - i)
            if(ch.isalpha()):
                return given_number - i
        else:
            if(is_prime(given_number + i)):
                ch = chr(given_number + i)
                if (ch.isalpha()):
                    return given_number + i
                    
print(find_nearest_prime(65))                    

