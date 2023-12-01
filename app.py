import pandas as pd
import streamlit as st
import pickle
model = pickle.load(open("https://github.com/phuongnhi201104/doancuoiky-/blob/main/model.sav", "rb"))
def predict(input_data):
    # Preprocess the input data
    # ...
    
    # Make predictions using the loaded model
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[:, 1]
    
    return predictions, probabilities
def main():
    # Set up the Streamlit app
    
    # Add input fields for user queries
    input_query1 = st.text_input("SeniorCitizen")
    input_query2 = st.text_input("MonthlyCharges")
    input_query3 = st.text_input("TotalCharges")
    # Add more input fields for the remaining features
    
    # Create a button to trigger the prediction
    if st.button("Predict"):
        # Create a DataFrame from the user inputs
        input_data = pd.DataFrame([[input_query1, input_query2, input_query3, ...]], 
                                  columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', ...])
        
        # Make predictions
        predictions, probabilities = predict(input_data)
        
        # Display the prediction results
        if predictions[0] == 1:
            st.write("This customer is likely to be churned!!")
        else:
            st.write("This customer is likely to continue!!")
        
        st.write("Confidence: {}%".format(probabilities[0] * 100))

if __name__ == "__main__":
    main()
app.run() 
