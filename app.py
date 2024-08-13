# https://youtu.be/l3QVYnMD128
"""
Application that predicts which sport your body was meant for. 

Trained on the data set of actual olympic athletes and their accomplishments in the field. 
The model uses their height, weight, age, sex, sport, team, year, 
and season in which the athelte competed to predict one's likelyhood of making it into that sport.
 The dataset is composed of 263785 observations of which 72,000 are used.

"""


import numpy as np
from flask import Flask, request, render_template
import pickle

#Create an app object using the Flask class. 
app = Flask(__name__)

#Load the trained model. (Pickle file)
model = pickle.load(open('models/model_dt.pkl', 'rb'))

#Define the route to be home. 
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, home function is with '/', our root directory. 
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder. 

#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')

#You can use the methods argument of the route() decorator to handle different HTTP methods.
#GET: A GET message is send, and the server returns data
#POST: Used to send HTML form data to the server.
#Add Post method to the decorator to allow for form submission. 
#Redirect to /predict page with the output
@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()] #Convert string inputs to float.
    sports = ['Basketball', 'Judo', 'Badminton', 'Sailing', 'Gymnastics', 'Wrestling', 'Rowing',
              'Athletics', 'Swimming', 'Football', 'Equestrianism', 'Shooting', 'Taekwondo',
              'Boxing', 'Weightlifting', 'Fencing', 'Diving', 'Canoeing', 'Handball', 'Water Polo',
              'Tennis', 'Cycling', 'Hockey', 'Softball', 'Volleyball', 'Synchronized Swimming', 
              'Modern Pentathlon', 'Table Tennis', 'Baseball', 'Rhythmic Gymnastics', 'Rugby Sevens', 
              'Archery', 'Trampolining', 'Beach Volleyball', 'Ice Hockey', 'Figure Skating']
    

    try: #Converts height from ft to cm
        tempHeight = str(int_features[0])
        tempHeight = ((float(tempHeight[0])*12) + float(tempHeight[1:])) * 2.54 
    except(ValueError):
        tempHeight += 0

    sex = int_features[3]
    age = int_features[2]
    height = tempHeight
    weight = (int_features[1])/(2.205)
    team = int_features[5]
    season = int_features[4]

    features = [np.array([sex, age, height, weight, team, season])]  #Convert to the form [[a, b, c, ...]] for input to the model
    print(features)
    prediction = model.predict(features)  # features Must be in the form [a, b, c, ...]
    output = round(prediction[0], 2)

    print(output)
    return render_template('results_page.html', sport_to_play ='Your Body Was Meant For  {}'.format(sports[int(output)]))


#When the Python interpreter reads a source file, it first defines a few special variables. 
#For now, we care about the __name__ variable.
#If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__). 
#So if we want to run our code right here, we can check if __name__ == __main__
#if so, execute it here. 
#If we import this file (module) to another file then __name__ == app (which is the name of this python file).

if __name__ == "__main__":
    app.run()