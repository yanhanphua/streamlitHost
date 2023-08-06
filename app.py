import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score
df = pd.read_csv("./Data/steam.csv")
genres_set = set()
for i in df.genres.str.split(';'):
    genres_set.update(i)

d = dict()

genre_sets = df.genres.str.split(';').apply(set)
for genre in genres_set:
    d[genre] = genre_sets.apply(lambda row: genre in row)
df = df.assign(**d)
platform_set = set()
for i in df.platforms.str.split(';'):
    platform_set.update(i)

platform_sets = df.platforms.str.split(';').apply(set)
d = dict()
d['windows'] = platform_sets.apply(lambda row: 'windows' in row)
d['linux'] = platform_sets.apply(lambda row: 'linux' in row)
d['mac'] = platform_sets.apply(lambda row: 'mac' in row)
df = df.assign(**d)

owners_mapping = {
    '100000000-200000000': 75000000,
    '50000000-100000000': 75000000,
    '20000000-50000000': 35000000,
    '10000000-20000000': 15000000,
    '5000000-10000000': 7500000,
    '2000000-5000000': 3500000,
    '1000000-2000000': 1500000,
    '500000-1000000': 750000,
    '200000-500000': 350000,
    '100000-200000': 125000,
    '50000-100000': 75000,
    '20000-50000': 35000,
    '0-20000': 10000
}

df['owners Edited'] = df['owners'].replace(owners_mapping)
replacement_mapping = {True: 1, False: 0}

# List of columns to be replaced
columns_to_replace = ['Strategy', 'Nudity', 'Action', 'Software Training',
                      'Early Access', 'Animation & Modeling', 'Web Publishing', 'Utilities',
                      'Accounting', 'Documentary', 'Adventure', 'Sexual Content', 'Racing',
                      'Sports', 'Violent', 'Gore', 'Video Production', 'Tutorial',
                      'Free to Play', 'Simulation', 'RPG', 'Game Development', 'Casual',
                      'Photo Editing', 'Audio Production', 'Indie', 'Massively Multiplayer',
                      'Design & Illustration', 'Education', 'windows', 'linux', 'mac']

# Replace boolean values in the specified columns
df[columns_to_replace] = df[columns_to_replace].replace(replacement_mapping)
machine_learning_dataframe= df.copy()
columns_to_drop = ['appid','name','release_date','developer','publisher','platforms','categories','genres','steamspy_tags','achievements','owners']
machine_learning_dataframe.drop(columns=columns_to_drop, inplace=True)
X = machine_learning_dataframe.drop('owners Edited',axis=1)
y = machine_learning_dataframe['owners Edited']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
clf_forest = RandomForestClassifier(n_estimators=100)
clf_forest.fit(X_train, y_train)
feature_names = X_train.columns.tolist()
user_input = {feature_name: 0 for feature_name in feature_names}
# Set up the Streamlit app
st.title("Steam Owners Classification")
st.write("Enter the dimensions of an steam games to classify its owners.")

# Collect user input
english = st.sidebar.selectbox('english', [0, 1], index=0)
required_age = st.sidebar.slider('Required Age', 0, 18, 0)
positive_ratings = st.sidebar.number_input('Positive Ratings', min_value=0, step=1, value=0)
negative_ratings = st.sidebar.number_input('Negative Ratings', min_value=0, step=1, value=0)
average_playtime = st.sidebar.number_input('Average Playtime', min_value=0, step=1, value=0)
median_playtime = st.sidebar.number_input('Median Playtime', min_value=0, step=1, value=0)
price = st.sidebar.number_input('Price', min_value=0.0, step=0.01, value=0.0)
software_training = st.sidebar.selectbox('Software Training', [0, 1], index=0)
video_production = st.sidebar.selectbox('Video Production', [0, 1], index=0)
massively_multiplayer = st.sidebar.selectbox('Massively Multiplayer', [0, 1], index=0)
education = st.sidebar.selectbox('Education', [0, 1], index=0)
nudity = st.sidebar.selectbox('Nudity', [0, 1], index=0)
photo_editing = st.sidebar.selectbox('Photo Editing', [0, 1], index=0)
violent = st.sidebar.selectbox('Violent', [0, 1], index=0)
rpg = st.sidebar.selectbox('RPG', [0, 1], index=0)
sports = st.sidebar.selectbox('Sports', [0, 1], index=0)
racing = st.sidebar.selectbox('Racing', [0, 1], index=0)
adventure = st.sidebar.selectbox('Adventure', [0, 1], index=0)
design_illustration = st.sidebar.selectbox('Design & Illustration', [0, 1], index=0)
game_development = st.sidebar.selectbox('Game Development', [0, 1], index=0)
accounting = st.sidebar.selectbox('Accounting', [0, 1], index=0)
web_publishing = st.sidebar.selectbox('Web Publishing', [0, 1], index=0)
simulation = st.sidebar.selectbox('Simulation', [0, 1], index=0)
action = st.sidebar.selectbox('Action', [0, 1], index=0)
early_access = st.sidebar.selectbox('Early Access', [0, 1], index=0)
casual = st.sidebar.selectbox('Casual', [0, 1], index=0)
audio_production = st.sidebar.selectbox('Audio Production', [0, 1], index=0)
sexual_content = st.sidebar.selectbox('Sexual Content', [0, 1], index=0)
animation_modeling = st.sidebar.selectbox('Animation & Modeling', [0, 1], index=0)
indie = st.sidebar.selectbox('Indie', [0, 1], index=0)
utilities = st.sidebar.selectbox('Utilities', [0, 1], index=0)
free_to_play = st.sidebar.selectbox('Free to Play', [0, 1], index=0)
tutorial = st.sidebar.selectbox('Tutorial', [0, 1], index=0)
gore = st.sidebar.selectbox('Gore', [0, 1], index=0)
strategy = st.sidebar.selectbox('Strategy', [0, 1], index=0)
documentary = st.sidebar.selectbox('Documentary', [0, 1], index=0)
windows = st.sidebar.selectbox('Windows', [0, 1], index=0)
linux = st.sidebar.selectbox('Linux', [0, 1], index=0)
mac = st.sidebar.selectbox('Mac', [0, 1], index=0)
classify_button = st.button("Classify")
if classify_button:
    user_input['english'] = [english]
    user_input['required_age'] = [required_age]
    user_input['positive_ratings'] = [positive_ratings]
    user_input['negative_ratings'] = [negative_ratings]
    user_input['average_playtime'] = [average_playtime]
    user_input['median_playtime'] = [median_playtime]
    user_input['price'] = [price]
    user_input['Nudity'] = [nudity]
    user_input['Casual'] = [casual]
    user_input['Web Publishing'] = [web_publishing]
    user_input['Massively Multiplayer'] = [massively_multiplayer]
    user_input['Racing'] = [racing]
    user_input['Sexual Content'] = [sexual_content]
    user_input['Simulation'] = [simulation]
    user_input['Game Development'] = [game_development]
    user_input['Tutorial'] = [tutorial]
    user_input['Education'] = [education]
    user_input['Sports'] = [sports]
    user_input['Adventure'] = [adventure]
    user_input['Action'] = [action]
    user_input['Strategy'] = [strategy]
    user_input['Design & Illustration'] = [design_illustration]
    user_input['Free to Play'] = [free_to_play]
    user_input['Indie'] = [indie]
    user_input['Documentary'] = [documentary]
    user_input['Utilities'] = [utilities]
    user_input['Gore'] = [gore]
    user_input['Audio Production'] = [audio_production]
    user_input['Video Production'] = [video_production]
    user_input['RPG'] = [rpg]
    user_input['Animation & Modeling'] = [animation_modeling]
    user_input['Photo Editing'] = [photo_editing]
    user_input['Early Access'] = [early_access]
    user_input['Violent'] = [violent]
    user_input['Software Training'] = [software_training]
    user_input['Accounting'] = [accounting]
    user_input['windows'] = [windows]
    user_input['linux'] = [linux]
    user_input['mac'] = [mac]
    
    # Convert the user input dictionary to a DataFrame
    user_input_df = pd.DataFrame(user_input)
    user_input_df
    # Make the prediction using the trained model
    predictions = clf_forest.predict(user_input_df)
    ans = f'the numbers of owners you can expect from a game like this is around {predictions}'
    ans
    # Display the prediction
    
