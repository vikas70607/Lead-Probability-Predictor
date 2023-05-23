import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Loading the Model 
loaded_model = pickle.load(open('Lead.pkl', 'rb'))

#loading the clean data
data = pd.read_csv('clean.csv')

def lead_score(input):
    cat = ['lost_reason','budget','duration','room_type']

    transformed_Array = []
    
    for i in range(0,4):
        le = LabelEncoder()
        le.fit(data[cat[i]]) 
        x = le.transform([input[i]])  
        transformed_Array.append(x[0])
    
    ans = loaded_model.predict_proba([transformed_Array])
    # ans is an array of probabilities where 0th element is 'LOST' probability and 1th element is 'WON' probability since we need the 'WON' probability we return first element of the array
    return ans[0][1]

#Example
check = lead_score(['Low budget','100','Semester Stay 20 - 24 weeks','Ensuite'])
print(check)    