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
  st.info('This app shows the best Straming Service you could subscribe to depending on your interests')
  
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
year_list = ['release_year_<1970','release_year_1970s', 'release_year_1980s', 'release_year_1990s', 
 'release_year_2000s', 'release_year_2010s', 'release_year_2020s']

year_selection = st.multiselect('Select release years', year_list)

# df_selection = df[df.genre.isin(genres_selection) & df['year'].isin(year_selection_list)]
# reshaped_df = df_selection.pivot_table(index='year', columns='genre', values='gross', aggfunc='sum', fill_value=0)
# reshaped_df = reshaped_df.sort_values(by='year', ascending=False)

type = st.selectbox("Select between Movie or Show", set(df["type"]))
duration = st.number_input(f"Duration of the {type} (minutes)", value=None)

if type == "SHOW":
  seasons = st.number_input("Number of seasons", value=None)
else:
  seasons = 0

# Train the model
X = df_train.drop(columns=["Streaming_Service"])
Y = df_train["Streaming_Service"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

model = RandomForestClassifier(n_estimators = 200, random_state=42)
model.fit(X_train, Y_train)

# Display DataFrame


df_editor = st.data_editor(reshaped_df, height=212, use_container_width=True,
                            column_config={"year": st.column_config.TextColumn("Year")},
                            num_rows="dynamic")
df_chart = pd.melt(df_editor.reset_index(), id_vars='year', var_name='genre', value_name='gross')

# Display chart
chart = alt.Chart(df_chart).mark_line().encode(
            x=alt.X('year:N', title='Year'),
            y=alt.Y('gross:Q', title='Gross earnings ($)'),
            color='genre:N'
            ).properties(height=320)
st.altair_chart(chart, use_container_width=True)
