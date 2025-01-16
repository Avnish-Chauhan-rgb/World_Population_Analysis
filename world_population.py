#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go


# In[2]:


from plotly.subplots import make_subplots
import warnings
# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
# Graph


# # IMPORT DATA

# In[3]:


df = pd.read_csv('C:/Users/Admin/Downloads/world_population.csv')
df.head()


# In[4]:


df.info()


# In[5]:


df.isna().sum()


# In[6]:


print(f"Amount of duplicates: {df.duplicated().sum()}")


# In[7]:


df.columns


# In[8]:


df.drop(['CCA3', 'Capital'], axis=1, inplace=True)


# In[9]:


df.head()


# In[10]:


df.tail()


# # Visualizations

# In[11]:


custom_palette = ['#0b3d91', '#e0f7fa', '#228b22', '#1e90ff', '#8B4513', '#D2691E',
'#DAA520', '#556B2F']


# In[12]:


countries_by_continent = df['Continent'].value_counts().reset_index()


# In[13]:


# Create the bar chart
fig = px.bar(
countries_by_continent,
x='Continent',
y='count',
color='Continent',
text='count',
title='Number of Countries by Continent',
color_discrete_sequence=custom_palette
)


# In[14]:


# Customize the layout
fig.update_layout(
xaxis_title='Continents',
yaxis_title='Number of Countries',
plot_bgcolor='rgba(0,0,0,0)', # Set the background color to transparent
font_family='Arial', # Set font family
title_font_size=20) # Set title font size
fig.show()


# In[15]:


continent_population_percentage = df.groupby('Continent')['World Population Percentage'].sum().reset_index()


# In[16]:


# Create the pie chart
fig = go.Figure(data=[go.Pie(labels=continent_population_percentage['Continent'],values=continent_population_percentage['World Population Percentage'])])

# Update layout
fig.update_layout(
title='World Population Percentage by Continent',
template='plotly',
paper_bgcolor='rgba(255,255,255,0)', # Set the paper background color to transparent
plot_bgcolor='rgba(255,255,255,0)' # Set the plot background color to transparent
)


# In[17]:


# Update pie colors
fig.update_traces(marker=dict(colors=custom_palette, line=dict(color='#FFFFFF',width=1)))
# Show the plot
fig.show()


# In[18]:


# Melt the DataFrame to have a long format
df_melted = df.melt(
    id_vars=['Continent'],
    value_vars=[
        '2022 Population', '2020 Population', '2015 Population',
        '2010 Population', '2000 Population', '1990 Population',
        '1980 Population', '1970 Population'
    ],
    var_name='Year',
    value_name='Population'
)

# Convert 'Year' to a more suitable format by extracting the year as an integer
df_melted['Year'] = df_melted['Year'].str.split().str[0].astype(int)

# Aggregate population by continent and year
population_by_continent = df_melted.groupby(['Continent', 'Year']).sum().reset_index()


# In[19]:


fig = px.line(population_by_continent, x='Year', y='Population', color='Continent',

title='Population Trends by Continent Over Time',
labels={'Population': 'Population', 'Year': 'Year'},
color_discrete_sequence=custom_palette)

fig.update_layout(template='plotly_white',
xaxis_title='Year',
yaxis_title='Population',
font_family='Arial',
title_font_size=20,
)

fig.update_traces(line=dict(width=3))

fig.show()


# # World Population Comparison: 1970 to 2020

# In[20]:


features=['1970 Population' ,'2020 Population']
for feature in features:
    fig = px.choropleth(df,

locations='Country/Territory',
locationmode='country names',
color=feature,
hover_name='Country/Territory',
template='plotly_white',
title = feature)

fig.show()


# In[21]:


growth = (df.groupby(by='Country/Territory')['2022 Population'].sum()-df.groupby(by='Country/Territory')['1970 Population'].sum()).sort_values(ascending=False).head(8)
                                             
fig=px.bar(x=growth.index,
y=growth.values,
text=growth.values,
color=growth.values,
title='Growth Of Population From 1970 to 2020 (Top 8)',
template='plotly_white')
fig.update_layout(xaxis_title='Country',

yaxis_title='Population Growth')

fig.show()                                             


# In[22]:


top_8_populated_countries_1970 = df.groupby('Country/Territory')['1970 Population'].sum().sort_values(ascending=False).head(8)
top_8_populated_countries_2022 = df.groupby('Country/Territory')['2022 Population'].sum().sort_values(ascending=False).head(8)

features = {'top_8_populated_countries_1970': top_8_populated_countries_1970, 'top_8_populated_countries_2022': top_8_populated_countries_2022}     

for feature_name, feature_data in features.items():
    year = feature_name.split('_')[-1] # Extract the year from the feature name
fig = px.bar(x=feature_data.index,
y=feature_data.values,
text=feature_data.values,
color=feature_data.values,
title=f'Top 8 Most Populated Countries ({year})',
template='plotly_white')
fig.update_layout(xaxis_title='Country',

yaxis_title='Population Growth')

fig.show()


# # World Population Growth Rates: The Fastest Growing Countries

# In[23]:


sorted_df_growth = df.sort_values(by='Growth Rate', ascending=False)
top_fastest = sorted_df_growth.head(6)
top_slowest = sorted_df_growth.tail(6)


# In[24]:


def plot_population_trends(countries):         # Calculate the number of rows needed
    n_cols = 2
    n_rows = (len(countries) + n_cols - 1) // n_cols

# Create subplots
    fig = sp.make_subplots(rows=n_rows, cols=n_cols, subplot_titles=countries,
    horizontal_spacing=0.1, vertical_spacing=0.1)
    
    for i, country in enumerate(countries, start=1):      # Filter data for the selected country
        country_df = df[df['Country/Territory'] == country]

# Melt the DataFrame to have a long format
        country_melted = country_df.melt(id_vars=['Country/Territory'],

        value_vars=['2022 Population', '2020 Population', '2015 Population', '2010 Population', '2000 Population', '1990 Population', '1980 Population', '1970 Population'],

        var_name='Year',
        value_name='Population')
    
# Convert 'Year' to a more suitable format
        country_melted['Year'] = country_melted['Year'].str.split().str[0].astype(int)
# Create a line plot for each country
        line_fig = px.line(country_melted, x='Year', y='Population', color='Country/Territory', labels={'Population': 'Population', 'Year': 'Year'}, color_discrete_sequence=custom_palette)
# Update the line plot to fit the subplot
        row = (i - 1) // n_cols + 1
        col = (i - 1) % n_cols + 1
        for trace in line_fig.data:
            fig.add_trace(trace, row=row, col=col)
# Update the layout of the subplots
    fig.update_layout(
    title='Population Trends of Selected Countries Over Time',
    template='plotly_white',
    font_family='Arial',
    title_font_size=20,
    showlegend=False,
    height=600*n_rows, # Adjust height for bigger plots
    )     
    fig.update_traces(line=dict(width=3))
    fig.update_xaxes(title_text='Year')
    fig.update_yaxes(title_text='Population')

    fig.show()


# In[ ]:





# In[25]:


fastest = top_fastest[['Country/Territory', 'Growth Rate']].sort_values(by='Growth Rate', ascending=False).reset_index(drop=True)
fastest


# In[26]:


plot_population_trends(['Moldova', 'Poland', 'Niger', 'Syria', 'Slovakia', 'DR Congo'])


# # World Population Growth Rates: The Slowest Growing Countries

# In[27]:


slowest = top_slowest[['Country/Territory', 'Growth Rate']].sort_values(by='Growth Rate', ascending=False).reset_index(drop=True)
slowest


# In[28]:


plot_population_trends(['Latvia', 'Lithuania', 'Bulgaria', 'American Samoa', 'Lebanon', 'Ukraine'])


# # Land Area by Country

# In[29]:


land_by_country = df.groupby('Country/Territory')['Area (km²)'].sum().sort_values(ascending=False)
most_land = land_by_country.head(5)
least_land = land_by_country.tail(5)


# In[30]:


# Create subplots
fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Countries with Most Land", "Countries with Least Land"))


# In[31]:


# Plot countries with the most land
fig.add_trace(go.Bar(x=most_land.index, y=most_land.values, name='Most Land',  marker_color=custom_palette[0]), row=1, col=1)


# In[32]:


# Plot countries with the least land
fig.add_trace(go.Bar(x=least_land.index, y=least_land.values, name='Least Land', marker_color=custom_palette[1]), row=1, col=2)


# In[33]:


fig.update_layout(
title_text="Geographical Distribution of Land Area by Country",
showlegend=False,
template='plotly_white'
)

fig.update_yaxes(title_text="Area (km2)", row=1, col=1)
fig.update_yaxes(title_text="Area (km2)", row=1, col=2)

fig.show()


# # Land Area Per Person by Country

# In[34]:


df['Area per Person']=df['Area (km²)'] / df['2022 Population']
country_area_per_person = df.groupby('Country/Territory')['Area per Person'].sum()
most_land_available = country_area_per_person.sort_values(ascending=False).head(5)
least_land_available = country_area_per_person.sort_values(ascending=False).tail(5)


# In[35]:


# Create subplots
fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Countries with Most Land Available Per Capita", "Countries with Least Land Available Per Capita"))

# Plot countries with the most land
fig.add_trace(go.Bar(x=most_land_available.index, y=most_land_available.values,
name='Most Land', marker_color=custom_palette[2]), row=1, col=1)

# Plot countries with the least land
fig.add_trace(go.Bar(x=least_land_available.index, y=least_land_available.values,
name='Least Land', marker_color=custom_palette[3]), row=1, col=2)

fig.update_layout(
title_text="Distribution of Available Land Area by Country Per Capita",
showlegend=False,
template='plotly_white'
)

fig.update_yaxes(title_text="Land Available Per Person", row=1, col=1)
fig.update_yaxes(title_text="Land Available Per Person", row=1, col=2)

fig.show()


# # Build Predective Model, Model Evaluation and Model Visualizations

# In[36]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
data = pd.read_csv('C:/Users/Admin/Downloads/world_population.csv')  # Adjust the file name

# Aggregate population by year for the entire world
world_population = {
    'Year': [1970, 1980, 1990, 2000, 2010, 2015, 2020, 2022],
    'Population': [
        data['1970 Population'].sum(),
        data['1980 Population'].sum(),
        data['1990 Population'].sum(),
        data['2000 Population'].sum(),
        data['2010 Population'].sum(),
        data['2015 Population'].sum(),
        data['2020 Population'].sum(),
        data['2022 Population'].sum(),
    ]
}

# Convert to a DataFrame
world_population_df = pd.DataFrame(world_population)

# Prepare the data for linear regression
X = world_population_df['Year'].values.reshape(-1, 1)  # Year as the independent variable
y = world_population_df['Population'].values  # World population as the dependent variable

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict future population
future_years = np.array([2025, 2030, 2035, 2040, 2050]).reshape(-1, 1)  # Years to predict
predictions = model.predict(future_years)

# Evaluate the model
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

# Output the results
print(f"Predictions for future years: {dict(zip(future_years.flatten(), predictions))}")
print(f"R-squared: {r2}")
print(f"Mean Absolute Error: {mae}")


# In[37]:


plt.figure(figsize=(10, 6))

# Plot historical population data
plt.scatter(world_population_df['Year'], world_population_df['Population'], color='blue', label='Actual Population')

# Plot the linear regression line
plt.plot(world_population_df['Year'], model.predict(X), color='red', linestyle='-', label='Linear Regression Line')

# Plot future predictions
plt.plot(future_years, predictions, color='green', marker='o', linestyle='--', label='Predicted Population')

# Labels and title
plt.title('World Population Over Time and Predictions', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Population (in billions)', fontsize=12)
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




