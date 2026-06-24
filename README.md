# Cancer Patient Status Prediction App-
This is a web application that analyzes clinical patient metrics to predict their current status or survival outcomes. I built the backend logic and the predictive machine learning model in Python, and used Streamlit to create the interactive web interface. 
The app is fully deployed and hosted live on the cloud.
# 🔗 Live Link-
You can test out the live app and check patient status predictions here:
[View Live Web App](https://cancer-data-analysis-oqelek8hz8bv2yb27rgke6.streamlit.app/)
# 🛠️ How it was built-
* Python — The main programming language used.
*  Scikit-Learn — Used to train a Random Forest classifier model on patient data records.
*  Streamlit — Used to build the web layout and inputs.
*  Joblib — Used to load the trained model weights (`cancer_rf_model.pkl`) directly into the app.
# 📁 Files in this project-
* "app.py": Contains the main code for the layout and prediction logic.
* "cancer_rf_model.pkl": The saved machine learning model.
* "requirements.txt": Lists the libraries needed to run the app on the web.
