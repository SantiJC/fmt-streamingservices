import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# Page title
st.set_page_config(page_title='Streaming Service Classifier', page_icon='📊')
st.title('📊 Streaming Service Classifier')

with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app shows the best Straming Service you could subscribe to depending on your interests')
  
st.subheader('Which Movie Genres you like most?')

# Load data
df = pd.read_csv('data/Data_No_Outliers.csv')


# Input widgets
## Genres selection
genres_list = ['action', 'animation', 'comedy', 'crime', 'documentation', 'drama', 
 'european', 'family', 'fantasy', 'history', 'horror', 'music', 'reality', 
 'romance', 'scifi', 'sport', 'thriller', 'war', 'western']
genres_selection = st.multiselect('Select genres', genres_list)

## Year selection
year_list = df.year.unique()
year_selection = st.slider('Select year duration', 1986, 2006, (2000, 2016))
year_selection_list = list(np.arange(year_selection[0], year_selection[1]+1))

df_selection = df[df.genre.isin(genres_selection) & df['year'].isin(year_selection_list)]
reshaped_df = df_selection.pivot_table(index='year', columns='genre', values='gross', aggfunc='sum', fill_value=0)
reshaped_df = reshaped_df.sort_values(by='year', ascending=False)


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
