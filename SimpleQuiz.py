
# coding: utf-8

# In[2]:


import random
continue_flag = True 

while (continue_flag):
    try:
        difficulty = input("Please choose the level of difficulty (easy(1), intermediate(2), hard(3)): ")
        difficulty = int(difficulty)
        if(int(difficulty) != 1 and int(difficulty) != 2 and int(difficulty) != 3):
            raise Exception
    except Exception:
        print("Not valid input for difficulty level")
        exit()
        
    num_questions = int(input("Please enter the number of questions you would like on the quiz: "))
    question_type = input("Specify the question type (multiplication:M, addition:A, subtraction:S, division:D): ")

    if(question_type == "M" or question_type == "m"):
        operator = "Multiplied by"
    elif(question_type == "A" or question_type == "a"):
        operator = "Added to"
    elif(question_type == "S" or question_type == "s"):
        operator = "Subtracted by"
    elif(question_type == "D" or question_type == "d"):
        operator = "Divided by"
    
    for i in range(num_questions):
        if(int(difficulty) == 1):
            op1 = random.randint(1, 10)
            op2 = random.randint(1, 10)
        elif(int(difficulty) == 2):
            op1 = random.randint(1, 100)
            op2 = random.randint(1, 100) 
        elif(difficulty == 3):
            op1 = random.randint(1, 1000)
            op2 = random.randint(1, 1000)
         
        user_answer = float(input("What is %d %s %d: " %(op1, operator, op2)))
        if (operator == "Multiplied by"):
            answer = op1 * op2
        elif (operator == "Added to"):
            answer = op1 + op2
        elif (operator == "Subtracted by"):
            answer = op1 - op2
        elif (operator == "Divided by"):
            answer = op1 / op2
        if(user_answer == answer):
            print("That's right! Well done.")
        else:
            print("Not quite right. The answer is: ", answer)
            
    continue_quiz = input("Continue or exit (Continue:C, Exit: E): ")
    if(continue_quiz.upper() == "C" ):
        continue_flag = True
    else:
        continue_flag = False
        
        
        
    

