#MINI PROJECT 1

#NAME - AYUSH KUMAR SHARMA
#YEAR - 2ND  DEPT - ECE
#email - sharmaayushKv@gmail.com
#KALYANI GOVERNMENT ENGINEERING COLLEGE

#PERFORMING EDA ON TITANIC DATASET


#PROGRAM PERFORMING EDA ON TITANIC DATASAET
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt




#TAKING THE TITANIC DATA
df = pd.read_csv(r"https://raw.githubusercontent.com/ameenmanna8824/DATASETS/main/train.csv")

#PRINTING THE DATASET

print('TITANIC DATASET'.center(120,'-'))
print(df)
print('-'.center(120,'-'))


#PERFORMING EDA

#EDA 1 :-
#GRAPH SHOWING THE SURVIVOR AND NON SURVIVOR UNDER AGE 20


new = df.where(df['Age']<20) #groupng data under the age of 20

sns.countplot(x='Survived',data=new)
plt.title('BAR GRAPH FOR SURVIVOR UNDER AGE OF 20')

plt.show()


#EDA 2 :-

print('DATA FOR SURVIVOR FOR PEOPLE UNDER AGE 20:-\n',new.groupby('Survived').size())
#hence , we conclude that 85 young kids died and 79 young kids survived
print('-'.center(120,'-'))


#EDA 3 :-

print('DATA FOR SURVIVOR FOR MALE AND FEMALE UNDER AGE 20:-\n',new.groupby(['Survived','Sex']).size())
#hence , we conclude 22 young female died and 63 young male died
#whereas 53 young female survived and 26 young male survived
print('-'.center(120,'-'))


#EDA 4 :-
#minimun price of fare

print('THE MINIMUM PRICE OF FARE :-',np.min(df['Fare']))
#so, according to our dataset some people payed no price for fare
print('-'.center(120,'-'))


#EDA 5 :-
#maximum price of fare

print('THE MAXIMUM PRICE OF FARE :-',np.max(df['Fare']))
#so, according to the dataset maximum price of 512.3292 was paid for fare
print('-'.center(120,'-'))


#EDA 6 :-
#BARGRAPH FOR SURVIVOR FROM DIFFRENT BOARDING LOCATION

sns.countplot(data=new,x= 'Embarked', hue= 'Survived')
plt.title('BAR GRAPH FOR SURVIVOR FROM BOARDING LOCATION')#S-Southampton, C- Cherbourg, Q - Queenstown
plt.show()
#HENCE , WE CONCLUDE THAT MOST PEOPLE DIED WHO WERE BOARDED FROM SOUTHAMPTON


#EDA 7 :-
#BARGRAPH FOR SURVIVOR FROM DIFFERENT CLASS

sns.countplot(data=new,x= 'Pclass', hue= 'Survived')
plt.title('BAR GRAPH FOR SURVIVOR FROM DIFFERENT CLASS')
plt.show()
#HENCE , FROM THIS GRAPH WE CONCLUDE THAT MOST PEOPLE THAT DIED IN TINATIC WERE FROM 3RD CLASS