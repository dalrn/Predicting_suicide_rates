import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import plotly.graph_objs as pgo

df = pd.read_csv('Suiciderate.csv')

st.title(':red[LIFESENSE]: The State of Worldwide Suicide, at your fingertips.')
st.markdown('Created by :red[Andalan Raihad] as a part of Arkalogica Competition ARKAVIDIA 8.0. ')

st.subheader('The dataset (Suicide Rate per 1000 Inhabitants):')


#FILTER MALE, FEMALE, BOTH
gender = st.selectbox('Filter by:', ('All', 'Male', 'Female', 'Both'), key=0)

df.replace([' Both sexes', ' Male', ' Female'], ['Both sexes', 'Male', 'Female'], inplace = True)
df['Sex'].unique()
df_male = df[df['Sex'] == 'Male']
df_female = df[df['Sex'] == 'Female']
df_both = df[df['Sex'] == 'Both sexes']

if gender == 'All': st.write(df)
if gender == 'Male': st.write(df_male)
if gender == 'Female': st.write(df_female)
if gender == 'Both': st.write(df_both)

st.caption('Dataset borrowed from [Kaggle - Mental Health and Suicide Rates](https://www.kaggle.com/datasets/twinkle0705/mental-health-and-suicide-rates)')
    
#SUICIDE RATE THROUGHOUT THE YEARS
total_both = df_both[['2000', '2010', '2015', '2016']].sum()
total_male = df_male[['2000', '2010', '2015', '2016']].sum()
total_female = df_female[['2000', '2010', '2015', '2016']].sum()

#make plot
st.title('')

st.subheader('Let\'s make it clearer with some :red[visualizations]\n\n')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('')

plt.figure(figsize=(10,6))
plt.title('Worldwide Suicide Rate Throughout the Years', fontdict={'fontweight' : 'bold', 'fontsize' : 18})
plt.xlabel('Year', fontdict={'fontweight' : 'bold', 'fontsize' : 13})
plt.ylabel('Suicide Rate', fontdict={'fontweight' : 'bold', 'fontsize' : 13})

plt.plot(total_both, marker = 'o', color = 'green')
plt.plot(total_male, marker = 'o', color = 'blue')
plt.plot(total_female, marker = 'o', color = 'red')

plt.legend(['Both', 'Male', 'Female'])

st.pyplot()

#WORLDWIDE SUICIDE PER YEAR

#TOP 10 COUNTRIES WITH HIGHEST SUICIDE RATE
df_male['Sum'] = df['2016']+df['2015']+df['2010']+df['2000']
df_male = df_male.sort_values('Sum', ascending=False)
top_male = list(df_male[:10].Country)
top_male_sum = list(df_male[:10].Sum)

df_female['Sum'] = df['2016']+df['2015']+df['2010']+df['2000']
df_female = df_female.sort_values('Sum', ascending=False)
top_female = list(df_female[:10].Country)
top_female_sum = list(df_female[:10].Sum)

df_both['Sum'] = df['2016']+df['2015']+df['2010']+df['2000']
df_both = df_both.sort_values('Sum', ascending=False)
top_both = list(df_both[:10].Country)
top_both_sum = list(df_both[:10].Sum)

#make bar chart
st.title('')
st.title('')

plt.figure(figsize=(25,10))
plt.title('Top 10 Countries With Highest Total Suicide Rate', fontdict={'fontweight' : 'bold', 'fontsize' : 18})
plt.xlabel('Country', fontdict={'fontweight' : 'bold', 'fontsize' : 13})
plt.ylabel('Suicide Rate', fontdict={'fontweight' : 'bold', 'fontsize' : 13})

maleplot = plt.bar(top_male, top_male_sum, color = 'blue')
bothplot = plt.bar(top_both, top_both_sum, color = 'green')
femaleplot = plt.bar(top_female, top_female_sum, color = 'red')

plt.legend(['Male', 'Both', 'Female'])

show_value = st.checkbox('Show value')
if show_value:
    plt.bar_label(maleplot,labels=top_male_sum)
    plt.bar_label(bothplot,labels=top_both_sum)
    plt.bar_label(femaleplot,labels=top_female_sum,label_type='center')

st.pyplot()


#HEATMAP VISUALIZATION
st.title('')
st.title('')

heat_year = st.selectbox('Year Input', ('2000', '2010', '2015', '2016'))

df_bothh = df[df['Sex'] == 'Both sexes']
df_bothh = df_bothh.reset_index()
df_bothh = df_bothh.drop(['index'], axis = 1)

df_heatmap = df_bothh[['Country', 'Sex', heat_year]]
df_2016=pd.melt(df_heatmap, id_vars=['Country','Sex'], var_name='Year', value_name='SR')

cloro= dict(type='choropleth',
            locations=df_2016['Country'],
            locationmode='country names',
            z=df_2016['SR'],
            text=df_2016['Country'],
            colorscale='Blues',
            reversescale=False,
            colorbar={'title':'Suicide Rate per 100.000 people'})

layout= dict(title= 'Suicide Rate Heatmap of Year ' + heat_year,
             geo= dict(showframe=True,
                       showcoastlines=True,
                      projection={'type':'miller'}))

choromap= pgo.Figure(data=[cloro],layout=layout)
st.write(choromap)


#PREDICTION APP
df['Sum'] = df['2016']+df['2015']+df['2010']+df['2000']

#modifying data
df_regression = df.sort_values('Sum', ascending=False)
df_both.reset_index(drop=True, inplace=True)

sex = pd.get_dummies(df['Sex'], drop_first = False)
df_regression = pd.concat([df, sex], axis = 1)
df_regression.drop(['Sex'], axis = 1, inplace  = True)

df_regression_melt = pd.melt(df_regression, id_vars = ['Country', 'Both sexes', 'Male', 'Female', 'Sum'], var_name = 'Year', value_name = 'Suicide Rate')
df_regression_melt.drop(columns = 'Sum', inplace = True)

le = sklearn.preprocessing.LabelEncoder() 
categorical = ['Country', 'Year']
for column in categorical:
    df_regression_melt[column] = le.fit_transform(df_regression_melt[column])
df_regression_melt_encoded = df_regression_melt

#split train and test
x = df_regression_melt.drop('Suicide Rate', axis = 1)
y = df_regression_melt['Suicide Rate']
x_train, x_test, y_train, y_test =  train_test_split(x, y, random_state = 42)

#train the model: XGBRegressor
xgb = XGBRegressor(learning_rate=0.2, max_depth=4)
xgb.fit(x_train, y_train)

y_test_xgb = xgb.predict(x_test)
y_train_xgb = xgb.predict(x_train)

acc_train_xgb = xgb.score(x_train, y_train)
acc_test_xgb = xgb.score(x_test, y_test)
rmse_train_xgb = np.sqrt(mean_squared_error(y_train, y_train_xgb))
rmse_test_xgb = np.sqrt(mean_squared_error(y_test, y_test_xgb))

#display result
st.markdown('**Let\'s see how it would be like in the :red[future], with our...**')
st.title('Suicide Rate Prediction App')

year = st.text_input('Year Input', value=None)
if year == 'None' or year == '':year = None
elif year.isdecimal():year = int(year)
else: st.error('Please enter a valid year.')

st.caption('Warning: The predictions will be less accurate the further to the future you try to look')
sort = st.selectbox('Sort by:', ('None', 'Highest', 'Lowest', 'Highest Increase', 'Lowest Increase', 'Highest Decrease', 'Lowest Decrease'))
filterby = st.selectbox('Filter by:', ('All', 'Male', 'Female', 'Both'), key=1)
search_bar = st.text_input("Search a country:")


future_data = pd.DataFrame({str(year)+' Prediction':y_test_xgb})
df1 = df[['Country', 'Sex']]
df2 = df.drop(['Country', 'Sex'], axis = 1)

result = pd.concat([df1, future_data, df2], axis = 1)
result = result.drop('Sum', axis = 1)
result['2016 - ' + str(year) + ' Difference'] = result[str(year)+' Prediction'] - result['2016']


#SORT BY INCREASED/DECREASED, FILTER BY GENDER, SEARCH FEATURE
result_highest = result.sort_values(str(year) + ' Prediction', ascending = False)
result_lowest = result.sort_values(str(year) + ' Prediction', ascending = True)

result_increase = result[result['2016 - ' + str(year) + ' Difference'] > 0]
result_increase_lowest = result_increase.sort_values('2016 - ' + str(year) + ' Difference', ascending = True)
result_increase_highest = result_increase.sort_values('2016 - ' + str(year) + ' Difference', ascending = False)

result_decrease = result[result['2016 - ' + str(year) + ' Difference'] < 0]
result_decrease_lowest = result_decrease.sort_values('2016 - ' + str(year) + ' Difference', ascending = False)
result_decrease_highest = result_decrease.sort_values('2016 - ' + str(year) + ' Difference', ascending = True)

if year != None:
    if sort == 'None':
        if filterby == 'Male':
            result_male = result.loc[result['Sex'] == 'Male']
            if search_bar:
                result_country = result_male[(search_bar == result_male.Country)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_male)
                    
        if filterby == 'Female':
            result_female = result.loc[result['Sex'] == 'Female']
            if search_bar:
                result_country = result_female[(result_female.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_female)

        if filterby == 'Both':
            result_both = result.loc[result['Sex'] == 'Both sexes']
            if search_bar:
                result_country = result_both[(search_bar in result_both['Country'])]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_both)

        if filterby == 'All':
            if search_bar:
                result_country = result[(result.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result)
    
    if sort == 'Highest':
        if filterby == 'Male':
            result_highest_male = result_highest.loc[result['Sex'] == 'Male']
            if search_bar:
                result_country = result_highest_male[(search_bar == result_highest_male.Country)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_highest_male)
                    
        if filterby == 'Female':
            result_highest_female = result_highest.loc[result['Sex'] == 'Female']
            if search_bar:
                result_country = result_highest_female[(result_highest_female.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_highest_female)

        if filterby == 'Both':
            result_highest_both = result_highest.loc[result['Sex'] == 'Both sexes']
            if search_bar:
                result_country = result_highest_both[(search_bar == result_highest_both.Country)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_highest_both)

        if filterby == 'All':
            if search_bar:
                result_country = result[(result.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_highest)
    
    if sort == 'Lowest':
        if filterby == 'Male':
            result_lowest_male = result_lowest.loc[result['Sex'] == 'Male']
            if search_bar:
                result_country = result_lowest_male[(search_bar == result_lowest_male.Country)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_lowest_male)
                    
        if filterby == 'Female':
            result_lowest_female = result_lowest.loc[result['Sex'] == 'Female']
            if search_bar:
                result_country = result_lowest_female[(result_lowest_female.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_lowest_female)

        if filterby == 'Both':
            result_lowest_both = result_lowest.loc[result['Sex'] == 'Both sexes']
            if search_bar:
                result_country = result_lowest_both[(result_lowest_both.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_lowest_both)

        if filterby == 'All':
            if search_bar:
                result_country = result[(result.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_lowest)
    
    if sort == 'Highest Increase':
        if filterby == 'Male':
            result_increase_highest_male = result_increase_highest.loc[result_increase_highest['Sex'] == 'Male']
            if search_bar:
                result_country = result_increase_highest_male[(result_increase_highest_male.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_increase_highest_male)
            
        if filterby == 'Female':
            result_increase_highest_female = result_increase_highest.loc[result_increase_highest['Sex'] == 'Female']
            if search_bar:
                result_country = result_increase_highest_female[(result_increase_highest_female.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_increase_highest_female)

        if filterby == 'Both':
            result_increase_highest_both = result_increase_highest.loc[result_increase_highest['Sex'] == 'Both sexes']
            if search_bar:
                result_country = result_increase_highest_both[(result_increase_highest_both.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_increase_highest_both)

        if filterby == 'All':
            if search_bar:
                result_country = result_increase_highest[(result_increase_highest.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_increase_highest)

    if sort == 'Lowest Increase':
        if filterby == 'Male':
            result_increase_lowest_male = result_increase_lowest.loc[result_increase_lowest['Sex'] == 'Male']
            if search_bar:
                result_country = result_increase_lowest_male[(result_increase_lowest_male.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_increase_lowest_male)

        if filterby == 'Female':
            result_increase_lowest_female = result_increase_lowest.loc[result_increase_lowest['Sex'] == 'Female']
            if search_bar:
                result_country = result_increase_lowest_female[(result_increase_lowest_female.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_increase_lowest_female)

        if filterby == 'Both':
            result_increase_lowest_both = result_increase_lowest.loc[result_increase_lowest['Sex'] == 'Both sexes']
            if search_bar:
                result_country = result_increase_lowest_both[(result_increase_lowest_both.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_increase_lowest_both)
            
        if filterby == 'All':
            if search_bar:
                result_country = result_increase_lowest[(result_increase_lowest.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_increase_lowest)

    if sort == 'Highest Decrease':
        if filterby == 'Male':
            result_decrease_highest_male = result_decrease_highest.loc[result_decrease_highest['Sex'] == 'Male']
            if search_bar:
                result_country = result_decrease_highest_male[(result_decrease_highest_male.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_decrease_highest_male)

        if filterby == 'Female':
            result_decrease_highest_female = result_decrease_highest.loc[result_decrease_highest['Sex'] == 'Female']
            if search_bar:
                result_country = result_decrease_highest_female[(result_decrease_highest_female.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_decrease_highest_female)

        if filterby == 'Both':
            result_decrease_highest_both = result_decrease_highest.loc[result_decrease_highest['Sex'] == 'Both sexes']
            if search_bar:
                result_country = result_decrease_highest_both[(result_decrease_highest_both.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_decrease_highest_both)

        if filterby == 'All':
            if search_bar:
                result_country = result_decrease_highest[(result_decrease_highest.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_decrease_highest)

    if sort == 'Lowest Decrease':
        if filterby == 'Male':
            result_decrease_lowest_male = result_decrease_lowest.loc[result_decrease_lowest['Sex'] == 'Male']
            if search_bar:
                result_country = result_decrease_lowest_male[(result_decrease_lowest_male.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_decrease_lowest_male)

        if filterby == 'Female':
            result_decrease_lowest_female = result_decrease_lowest.loc[result_decrease_lowest['Sex'] == 'Female']
            if search_bar:
                result_country =  result_decrease_lowest_female[( result_decrease_lowest_female.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write( result_decrease_lowest_female)

        if filterby == 'Both':
            result_decrease_lowest_both = result_decrease_lowest.loc[result_decrease_lowest['Sex'] == 'Both sexes']
            if search_bar:
                result_country = result_decrease_lowest_both[(result_decrease_lowest_both.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)    
            else: st.write(result_decrease_lowest_both)

        if filterby == 'All':
            if search_bar:
                result_country = result_decrease_lowest[(result_decrease_lowest.Country == search_bar)]
                if result_country.empty: st.text('No countries found')
                else: st.write(result_country)
            else: st.write(result_decrease_lowest)
else:
    st.text('Please input a year to predict')


st.write("Accuracy on training Data: {:.3f}".format(acc_train_xgb))
st.caption("Prediction accuracy: 0.798")
st.write('The RMSE of the training set is: ', rmse_train_xgb)
st.write('The RMSE of the testing set is: ', rmse_test_xgb)

log = np.log10(10)
print(log)