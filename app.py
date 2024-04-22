import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# Page Configs
st.set_page_config(page_title="Medical Insurance", page_icon="📊", layout='wide', initial_sidebar_state='expanded')

# Remove Default theme
theme_plotly = None

# CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Decision Tree Regression": DecisionTreeRegressor(),
    "Random Forest Regression": RandomForestRegressor(),
    "Gradient Boosting Regression": GradientBoostingRegressor(),
}

# Load data
@st.cache_data
def load_data(csv:str):
    df = pd.read_csv(csv)
    return df

# Prediction
def prediction(data, regression_model, user_data):
    categorical_features = ['sex', 'smoker', 'region']
    one_hot = OneHotEncoder()
    transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")
    X_transformed = transformer.fit_transform(data.drop('charges', axis=1))
    y = data['charges']
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
    data['charges_category'] = (data['charges'] > data['charges'].median()).astype(int)
    user_data_transformed = transformer.transform(user_data)
    models[regression_model].fit(X_train, y_train)
    predict_charges = models[regression_model].predict(user_data_transformed)[0]
    
    return predict_charges

# Taking user input to predict medical insurance cost
def user_input():
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    age = col1.number_input('Age: ', value=None)
    sex = col2.selectbox('Gender (male/female): ',
    ('male', 'female'))
    bmi = col3.number_input('BMI: ', value=None)
    children = col4.number_input('Number of children: ', value=None)
    smoker = col5.selectbox('Are you a smoker? (yes/no): ',
    ('yes', 'no'))
    region = col6.selectbox('Region ',
    ('southwest', 'southwest', 'northwest', 'northeast'))
    return age, sex, bmi, children, smoker, region

def on_button_click():
    st.session_state.clicked = True

def on_predict(age, sex, bmi, children, smoker, region, regression_model):
    col1, col2, col3, col4, col5 = st.columns(5)
    col3.button('Predict', on_click=on_button_click)
    if st.session_state.clicked:
        if age == None:
            col3.write('age is required')
        if bmi == None:
            col3.write('bmi is required')
        if children == None:
            col3.write('Number of children is required')
        else:
            user_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]], columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
            predict_charges = prediction(data, regression_model, user_data)
            card1, card2, card3 = st.columns(3)
            card2.metric('Predicted charges', f'${predict_charges:.2f}')

# Box plot for age, bmi
def box_plot(data):
    age_bmi = pd.DataFrame(zip(data['age'],data['bmi']), columns = ['age','bmi'])
    fig = px.box(age_bmi, title='Comparison between Age and BMI')
    fig.update_layout(yaxis_title='')
    fig.update_layout(xaxis_title='')
    return fig

# Scatter plot for smoker, charges
def scatter_plot(data):
    fig = px.scatter(x=data['smoker'], y=data['charges'], title='Smoker-Charges')
    fig.update_layout(yaxis_title='Charges')
    fig.update_layout(xaxis_title='Smoker')
    return fig

# Heat map
def heat_map(data):
    df_dummied = pd.concat([data, pd.get_dummies(data[['sex', 'smoker', 'region']])], axis=1)
    df_dummied.drop(columns=['sex', 'smoker', 'region'], inplace=True)
    fig = px.imshow(df_dummied.corr(), title='Primary factors influencing medical expenses', text_auto=True, aspect="auto")
    return fig


if __name__ == "__main__":
    # Getting the data
    data = load_data('medical_insurance.csv')

    # Main Page
    st.header('Medical Insurance cost EDA')
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False
    
    # Medical Insurance Prediction
    with st.expander('Predict you Medical Insurance cost'):
        age, sex, bmi, children, smoker, region = user_input()
        col1, col2, col3 = st.columns(3)
        with col2:
            regression_model = st.selectbox('Regression Model: ',
            ('Linear Regression', 'Ridge Regression', 'Decision Tree Regression', 'Random Forest Regression', 'Gradient Boosting Regression'))
            user_data = on_predict(age, sex, bmi, children, smoker, region, regression_model)
    
    # Visualizations
    plot1, plot2 = st.columns(2)
    plot1.plotly_chart(box_plot(data), use_container_width=True)
    plot2.plotly_chart(scatter_plot(data), use_container_width=True)
    st.plotly_chart(heat_map(data), use_container_width=True)

    st.markdown('---')
    col1, col2, col3,col4,col5 = st.columns(5)
    col3.markdown('Created by [Alif Hossain Sajib](https://github.com/AlifHossain27)')
    


    