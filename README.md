 Flight Price Prediction System

 Project Overview

This project is a Machine Learning-based application that predicts flight ticket prices using various input features such as airline, source, destination, duration, and number of stops. The system demonstrates a complete workflow from model training to deployment using Streamlit.

---

 Objective

The main objective of this project is to develop a predictive model that can estimate flight ticket prices accurately and provide users with a simple web interface to interact with the model.

 Dataset

The dataset used in this project contains flight-related information and is stored as `Clean_Dataset.csv`.

Features in Dataset:

* Airline: Name of the airline company
* Source: Departure city
* Destination: Arrival city
* Route: Path taken by the flight (including stops)
* Duration: Total travel time (e.g., 2h 50m)
* Total Stops: Number of stops (Non-stop, 1 stop, etc.)
* Additional Information: Extra flight details
* Price: Ticket price (Target Variable)

 Dataset Type:

* Supervised Learning Dataset
* Regression Problem (predicting continuous values)

 Preprocessing Steps:

* Handling missing values and duplicates
* Converting duration into numerical format
* Encoding categorical variables (Airline, Source, Destination, Stops)
* Feature selection and transformation

Note: The dataset may not be included due to size limitations. It is available within the training notebook or can be sourced externally.

 Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Streamlit
* Pickle


## Machine Learning Workflow

1. Data Collection and Loading
2. Data Cleaning and Preprocessing
3. Feature Engineering
4. Encoding Categorical Variables
5. Model Training using Machine Learning Algorithms
6. Model Evaluation and Selection
7. Model Saving using Pickle

Model Information

The model is trained using regression-based machine learning algorithms to predict flight prices. The trained model is saved as a `.pkl` file.

Note: The model file is not included in this repository due to GitHub file size limitations.

 How to Run the Project

 Step 1: Clone the Repository

```bash
git clone https://github.com/Vinay7780/Flight_-Predection.git
cd Flight_-Predection
```

 Step 2: Create Virtual Environment

python -m venv venv
venv\Scripts\activate
```

Step 3: Install Dependencies


pip install -r requirements.txt
```

Step 4: Generate the Model

Run the notebook file to train and generate the model:
model_training.ipynb

 Step 5: Run the Application


streamlit run app.py
```

---

 Project Structure

```
Flight_-Predection/
│
├── app.py
├── requirements.txt
├── model_training.ipynb
├── README.md
```
 Important Note

The trained model file (.pkl) is not uploaded due to GitHub file size limitations. To generate the model, please run the notebook (model_training.ipynb).
 Conclusion

This project demonstrates a complete end-to-end machine learning pipeline, including data preprocessing, model building, evaluation, and deployment using a web interface.

