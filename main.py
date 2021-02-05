import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title('National Basketball Association Player Stats Explorer')

st.markdown("""
This App performs simple webscraping of National Basketball Association Player Stats Data!
* **Python libraries:** base64, pandas, streamlit
* **Data source:**  [Basketball-reference.com](http://www.basketball-reference.com/).
""")

# Sidebar
# Header
st.sidebar.header('User Input Features')
# SelectBox 2021-1950
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950, 2021))))

# // Web Scraping Data source of National Basketball Association Player Stats


@st.cache
def load_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + \
        str(year) + "_per_game.html"
    html = pd.read_html(url, header=0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index)
    raw = raw.fillna(0)
    playerstats = raw.drop(['Rk'], axis=1)
    return playerstats


playerstats = load_data(selected_year)

# Sidebar - Select Team
sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect(
    'Team', sorted_unique_team, sorted_unique_team)

# Sidebar select Position
unique_pos = ['C', 'PF', 'SF', 'PG', 'SG']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

# Filtering data
df_selected_team = playerstats[(playerstats.Tm.isin(
    selected_team)) & (playerstats.Pos.isin(selected_pos))]

st.header('Display Player Stats pf Selected Team')
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(
    df_selected_team.shape[1]) + ' columns.')
st.dataframe(df_selected_team)

# Download Data


def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download Csv File</a>'
    return href


st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

# Heatmap

if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df_selected_team.to_csv('output.csv', index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot()
