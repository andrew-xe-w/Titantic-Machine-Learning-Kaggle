import numpy as np
import pandas as pd

def model(Class2, Class3, Male, Fare, SibSp):
    intercept = 2.208955
    beta1 = -0.660665
    beta2 = -1.647788
    beta3 = -2.740031
    beta4 = 0.003409
    beta5 = -0.278028

    odds = np.exp(intercept + beta1*Class2 + beta2*Class3 + beta3*Male + beta4*Fare + beta5*SibSp)
    probability = odds/(1 + odds)
    
    return probability



if __name__ == "__main__":
    test_dat = pd.read_csv("test.csv")
    df = pd.DataFrame(columns = ["PassengerId","Survived"])
    
    for index, row in test_dat.iterrows():
        Class2 = 0
        Class3 = 0
        Male = 0
        if row['Pclass'] == 2:
            Class2 = 1
        if row['Pclass'] == 3:
            Class3 = 1
        if row['Sex'] == "male":
            Male = 1

        p = model(Class2, Class3, Male, row['Fare'], row['SibSp'])

        df.loc[index] = [row['PassengerId'], round(p)]
        
    print(df)
    df.to_csv('regression_predictions.csv', index=False)
        

        