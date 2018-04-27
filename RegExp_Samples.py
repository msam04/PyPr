
# coding: utf-8

# In[24]:


import re
user_input = input("Please enter the string to parse: ")
res_date = re.match(r"(\d{4})-(\d{2})-(\d{2})", user_input)
if (res_date):
    print(res_date.group())
    print(res_date.group(1))
    print(res_date.group(2))
    print(res_date.group(3))
    print("The given input is a date.")
    
res_ssn = re.match("r(\d{3}-(\d{3})-(\d{4}))", user_input)    
if (res_ssn):
    print(res_ssn.group(1))
    print(res_ssn.group(2))
    print(res_ssn.group(3))
    print("The given input is an SSN.")
    
res_ip = re.match(r"^(([0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])$", user_input)
#res_ip = re.match(r"^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$", user_input)
    
if (res_ip):
    print(res_ip.group(1))
    print(res_ip.group(2))
    print("The given input is an IP address.")
    
res_email = re.match(r"\w+@\w+.\w+", user_input)
if res_email:
    print("The given input is an email address.")
    #print(res_email.group(2))
    
    #print(res.group(0), res.group(1), res.group(2))
 #   print("Inside if")
#if (re.match (r"^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5]))$", user_input)):


