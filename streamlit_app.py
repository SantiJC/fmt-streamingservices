import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Page title
st.set_page_config(page_title='Streaming Service Classifier', page_icon='📊')
st.title('📊 Streaming Service Classifier')

with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app shows the best Straming Service among Netflix, Prime Video, Hulu, Disney + and HBO MAX that you could subscribe to depending on your interests')
  
st.subheader('Which Movie Genres you like most?')

# Load data
df = pd.read_csv('data/Data_No_Outliers.csv')
df_train = pd.read_csv('data/df_train.csv')


# Input widgets
## Genres selection
genres_list = ['action', 'animation', 'comedy', 'crime', 'documentation', 'drama', 
 'european', 'family', 'fantasy', 'history', 'horror', 'music', 'reality', 
 'romance', 'scifi', 'sport', 'thriller', 'war', 'western']
genres_selection = st.multiselect('Select genres', genres_list)

## Year selection
year_list = sorted(set(df["release_year"]))

year_selection = st.multiselect('Select release years', year_list)

type = st.selectbox("Select between Movie or Show", set(df["type"]))

if type == "SHOW":
  show = 0
  selection = "shows"
elif type == "MOVIE":
  show = 1
  selection = "movies"

duration = st.number_input(f"Duration of the {type} (minutes)", value=None)

if type == "SHOW":
  seasons = st.number_input("Number of seasons", value=None)
else:
  seasons = 0

score = st.number_input(f"Score of the {type} (1 to 10)")


locations= ['América del Norte', 'América del Sur', 'África', 'Europa', 'Oceanía', 'Asia']

zone = st.multiselect('Select the production location', locations)



if '<1970' in year_selection:
  year1 = 1
else: 
  year1 = 0
if '1970s' in year_selection:
  year1970 = 1
else:
  year1970 = 0
if '1980s' in year_selection:
  year1980 = 1
else:
  year1980 = 0
if '1990s' in year_selection:
  year1990 = 1
else:
  year1990 = 0
if '2000s' in year_selection:
  year2000 = 1
else:
  year2000 = 0
if '2010s' in year_selection:
  year2010 = 1
else:
  year2010 = 0
if '2020s' in year_selection:
  year2020 = 1
else:
  year2020 = 0




if 'action' in genres_selection:
  action = 1
else:
  action = 0

if 'animation' in genres_selection:
  animation = 1
else:
  animation = 0

if 'comedy' in genres_selection:
  comedy = 1
else:
  comedy = 0

if 'crime' in genres_selection:
  crime = 1
else:
  crime = 0

if 'documentation' in genres_selection:
  documentation = 1
else:
  documentation = 0

if 'drama' in genres_selection:
  drama = 1
else:
  drama = 0

if 'european' in genres_selection:
  european = 1
else:
  european = 0

if 'family' in genres_selection:
  family = 1
else:
  family = 0

if 'fantasy' in genres_selection:
  fantasy = 1
else:
  fantasy = 0

if 'history' in genres_selection:
  history = 1
else:
  history = 0

if 'horror' in genres_selection:
  horror = 1
else:
  horror = 0

if 'music' in genres_selection:
  music = 1
else:
  music = 0

if 'reality' in genres_selection:
  reality = 1
else:
  reality = 0

if 'romance' in genres_selection:
  romance = 1
else:
  romance = 0

if 'scifi' in genres_selection:
  scifi = 1
else:
  scifi = 0

if 'sport' in genres_selection:
  sport = 1
else:
  sport = 0

if 'thriller' in genres_selection:
  thriller = 1
else:
  thriller = 0

if 'war' in genres_selection:
  war = 1
else:
  war = 0

if 'western' in genres_selection:
  western = 1
else:
  western = 0




if 'América del Norte' in zone:
    america_norte = 1
else:
    america_norte = 0

if 'América del Sur' in zone:
    america_sur = 1
else:
    america_sur = 0

if 'África' in zone:
    africa = 1
else:
    africa = 0

if 'Europa' in zone:
    europa = 1
else:
    europa = 0

if 'Oceanía' in zone:
    oceania = 1
else:
    oceania = 0

if 'Asia' in zone:
    asia = 1
else:
    asia = 0

# Train the model
# try:
X = df_train.drop(columns=["Streaming_Service"])
Y = df_train["Streaming_Service"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

model = RandomForestClassifier(n_estimators = 200, random_state=42)
model.fit(X_train, Y_train)
new_case = pd.DataFrame({
    "runtime": [duration],
    "seasons": [seasons],
    "imdb_score": [score],
    "action": [action],
    "animation": [animation],
    "comedy": [comedy],
    "crime": [crime],
    "documentation": [documentation],
    "drama": [drama],
    "european": [european],
    "family": [family],
    "fantasy": [fantasy],
    "history": [history],
    "horror": [horror],
    "music": [music],
    "reality": [reality],
    "romance": [romance],
    "scifi": [scifi],
    "sport": [sport],
    "thriller": [thriller],
    "war": [war],
    "western": [western],
    "América del Sur": [america_sur],
    "América del Norte": [america_norte],
    "África": [africa],
    "Europa": [europa],
    "Oceanía": [oceania],
    "Asia": [asia],
    "type_SHOW": [show],
    "release_year_1970s": [year1970],
    "release_year_1980s": [year1980],
    "release_year_1990s": [year1990],
    "release_year_2000s": [year2000],
    "release_year_2010s": [year2010],
    "release_year_2020s": [year2020],
    "release_year_<1970": [year1]
})


prediction = model.predict(new_case)
prediction = str(prediction).replace("['", "").replace("']", "")
st.divider()
st.markdown("<h2 style='text-align: left;'>Your recommended streaming service is:</h2>", unsafe_allow_html=True)
st.markdown(f"<h1 style='text-align: center;'>{prediction}</h1>", unsafe_allow_html=True)

results = []
for x in new_case.columns:
    y = int(new_case[x])
    if y != 0:
        results.append(y)
    elif y == 0:
        results.append(y) 
if results[28] == 1:
  movie_show = "MOVIE"
elif results[28] == 0:
  movie_show = "SHOW"

streaming_service_condition = df["Streaming_Service"] == prediction
type_show_condition = df["type"] == movie_show
columns_and_indices = [
  ("runtime", 0),
  ("seasons", 1),
  ("imdb_score", 2),
  ("action", 3),
  ("animation", 4),
  ("comedy", 5),
  ("crime", 6),
  ("documentation", 7),
  ("drama", 8),
  ("european", 9),
  ("family", 10),
  ("fantasy", 11),
  ("history", 12),
  ("horror", 13),
  ("music", 14),
  ("reality", 15),
  ("romance", 16),
  ("scifi", 17),
  ("sport", 18),
  ("thriller", 19),
  ("war", 20),
  ("western", 21),
  ("América del Sur", 22),
  ("América del Norte", 23),
  ("África", 24),
  ("Europa", 25),
  ("Oceanía", 26),
  ("Asia", 27),
  ("release_year_1970s", 29),
  ("release_year_1980s", 30),
  ("release_year_1990s", 31),
  ("release_year_2000s", 32),
  ("release_year_2010s", 33),
  ("release_year_2020s", 34),
  ("release_year_<1970", 35)
]
st.dataframe(df.head(5))
st.markdown(results)
additional_conditions = []
for column, index in columns_and_indices:
  if results[index] != 0:
      additional_conditions.append(df[column] == results[index])
if additional_conditions:
    combined_condition = additional_conditions[0]
    for condition in additional_conditions[1:]:
        combined_condition = combined_condition | condition
    final_condition = streaming_service_condition & type_show_condition & combined_condition
else:
  final_condition = streaming_service_condition & type_show_condition
df_predicted = df[final_condition]
df_predicted = df_predicted.drop(columns=["Index","imdb_id", "imdb_votes", "tmdb_popularity", "tmdb_score", 'action', 'animation', 'comedy', 'crime', 'documentation', 'drama', 
'european', 'family', 'fantasy', 'history', 'horror', 'music', 'reality', 
'romance', 'scifi', 'sport', 'thriller', 'war', 'western', 'age_certification', 'id', 'América del Norte', 'América del Sur', 'África', 'Europa', 'Oceanía', 'Asia', 'Antártida']).sort_values(by="imdb_score", ascending=False)

st.divider()
st.markdown(f"Top 20 best {selection} that match your preferences in {prediction} are:")
st.dataframe(df_predicted.head(20))
# except:
st.markdown("<h1 style='text-align: center;'>Complete the fields above</h1>", unsafe_allow_html=True)