import streamlit as st
from joblib import dump, load
import numpy as np

log_model = load('university_admission.joblib')

def classify(gre,tofel,sepal_length, sepal_width, petal_length, petal_width,research):
    inputs=np.array([[gre,tofel,sepal_length, sepal_width, petal_length, petal_width,research]]).reshape(1, -1)

    df = pd.read_csv("C:/Users/ubaid khan/Desktop/ML & DS/PROJECT/University Admission Probability/Admission_Predict.csv")
    df = df.rename(columns = {'Chance of Admit ':'Chance of Admit'})
    df = df.drop('Serial No.', axis=1)
    XX = df.drop('Chance of Admit', axis=1)

    XX.iloc[0] = [inputs]
    XX = scalerrr.fit_transform(XX)
    log_model.predict([XX[0]])

    from sklearn.preprocessing import StandardScaler
    scalerrr = StandardScaler()
    scalerrr.fit(inputs)
    inputs = scalerrr.transform(inputs)
    
    from sklearn.preprocessing import StandardScaler
    scalerr = StandardScaler()
    scalerr.fit(inputs)
    inputs = scalerr.transform(inputs)

    predictionn = log_model.predict(inputs)
    predd = '{}'.format(predictionn)
    return(float(predd[1:7])*100)

def main():
    st.title("University Acceptance Predictor")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    </div>
    """

    gre=st.number_input('GRE Score')

    tofel=st.number_input('TOFEL Score')

    st.markdown(html_temp, unsafe_allow_html=True)
    sepal_length=st.slider('University Rating', 0.0, 5.0)
    sepal_width=st.slider('SOP Rating', 0.0, 5.0)
    petal_length=st.slider('LOR Rating', 0.0, 5.0)
    petal_width=st.slider('Select CGPA', 0.0, 10.0)
    #research=st.slider('Select Research', 0.0, 1.0)

    R = st.radio('Did Research?', ('Yes','No'))
    if R == 'Yes':
        research = 1
    else:
        research = 0

    inputs=np.array([[gre,tofel,sepal_length, sepal_width, petal_length, petal_width,research]]).reshape(1, -1)
   

    if st.button('Predict My Chances'):
        output= classify(gre,tofel,sepal_length, sepal_width, petal_length, petal_width,research)
        st.success('Your chance of admission is: {}%'.format(output))


if __name__=='__main__':
    main()
