import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import pickle
import os

st.set_page_config(page_title="Personal Fitness Tracker", layout="wide")

# Function to train and evaluate models with hyperparameter tuning
def train_evaluate_models(X_train, X_test, y_train, y_test):
    # Define parameter grids for tuning
    param_grids = {
        "Random Forest Regressor": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20]
        },
        "Decision Tree Regressor": {
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10]
        },
        "Support Vector Regressor": {
            "C": [0.1, 1, 10],
            "epsilon": [0.01, 0.1, 1]
        },
        "Neural Network Regressor": {
            "hidden_layer_sizes": [(32,), (64, 32), (128, 64, 32)],
            "alpha": [0.0001, 0.001, 0.01]
        }
    }
    
    # Initialize models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Support Vector Regressor": SVR(),
        "Neural Network Regressor": MLPRegressor(max_iter=1000, random_state=42)
    }
    
    best_models = {}
    results = {}
    best_model = None
    best_score = -np.inf
    
    # Train and tune models
    for name, model in models.items():
        progress_text = st.text(f"Training {name}...")
        
        if name in param_grids:
            with st.spinner(f"Tuning {name} using GridSearchCV..."):
                grid_search = GridSearchCV(model, param_grids[name], scoring="neg_mean_squared_error", cv=5)
                grid_search.fit(X_train, y_train)
                best_models[name] = grid_search.best_estimator_
                st.write(f"Best Parameters for {name}: {grid_search.best_params_}")
        else:
            with st.spinner(f"Training {name}..."):
                model.fit(X_train, y_train)
                best_models[name] = model  # No tuning needed for Linear Regression
        
        # Make predictions and evaluate
        model = best_models[name]
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mse': mse
        }
        
        # Update progress
        progress_text.text(f"{name} trained. MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
        
        # Track the best model based on R² score
        if r2 > best_score:
            best_score = r2
            best_model = name
    
    return results, best_model, best_models

# Function to save model
def save_model(model, filename="calorie_model.pkl"):
    pickle.dump(model, open(filename, "wb"))
    return filename

# Main page layout
st.title("🔥 Personal Fitness Tracker 🔥")
st.markdown("Get insights on your fitness based on your personal parameters.")

# Upload files
st.sidebar.header("📁 Upload Data Files")
calories_file = st.sidebar.file_uploader("Upload Calories CSV", type=['csv'], key="calories")
exercise_file = st.sidebar.file_uploader("Upload Exercise CSV", type=['csv'], key="exercise")

# Check if both files are uploaded
if calories_file is not None and exercise_file is not None:
    # Load datasets
    calories = pd.read_csv(calories_file)
    exercise = pd.read_csv(exercise_file)
    
    # Display data samples
    with st.expander("📊 Sample Data"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Calories Data")
            st.write(calories.head())
        with col2:
            st.subheader("Exercise Data")
            st.write(exercise.head())
    
    # Merge datasets
    try:
        df = exercise.merge(calories, on="User_ID")
        if "User_ID" in df.columns:
            df.drop(columns="User_ID", inplace=True)
        
        # Add BMI column
        df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
        df["BMI"] = round(df["BMI"], 2)
        
        # Display exploratory data
        st.subheader("🔍 Exploratory Data Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df, x="Duration", y="Calories", color="Gender", 
                           title="Duration vs Calories", hover_data=["Age", "BMI"])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(df, x="Heart_Rate", y="Calories", color="Gender", 
                           title="Heart Rate vs Calories", hover_data=["Age", "BMI"])
            st.plotly_chart(fig, use_container_width=True)
        
        # Prepare data for modeling
        X = df[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp"]]
        y = df["Calories"]
        X = pd.get_dummies(X, drop_first=True)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        
        # Model Training Section
        st.subheader("🤖 Model Training")
        
        if st.button("Train Models"):
            # Train all models and get results
            results, best_model_name, best_models = train_evaluate_models(X_train, X_test, y_train, y_test)
            
            # Save the best model
            save_model(results[best_model_name]['model'])
            st.session_state['trained'] = True
            st.session_state['best_model_name'] = best_model_name
            st.session_state['model_results'] = results
            st.session_state['best_models'] = best_models
            st.session_state['X_columns'] = X.columns.tolist()
            
            st.success(f"Models trained successfully! Best model: {best_model_name}")
            
            # Display model comparison
            results_df = pd.DataFrame({
                'Model': list(results.keys()),
                'MAE': [results[model]['mae'] for model in results],
                'RMSE': [results[model]['rmse'] for model in results],
                'R² Score': [results[model]['r2'] for model in results],
                'MSE': [results[model]['mse'] for model in results]
            })
            
            st.write("Model Comparison:")
            st.write(results_df.sort_values('R² Score', ascending=False))
            
            # Visualize model comparison
            fig = px.bar(results_df, x='Model', y='R² Score', title="Model Performance Comparison",
                       color='Model', text='R² Score')
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            # Error comparison
            fig = px.bar(results_df, x='Model', y=['MAE', 'RMSE'], 
                       title="Error Metrics Comparison", barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        # User Input for Prediction
        if 'trained' in st.session_state and st.session_state['trained']:
            st.subheader("🔧 User Parameters")
            
            # Create columns for better layout of input fields
            col1, col2 = st.columns(2)
            
            # Column 1 inputs
            with col1:
                # Toggle for input method
                input_method = st.radio("Input Method", ["Sliders", "Direct Input"])
                
                if input_method == "Sliders":
                    age = st.slider("Age", 10, 100, 30)
                    height = st.slider("Height (cm)", 140, 200, 170)
                    weight = st.slider("Weight (kg)", 40, 120, 70)
                    
                else:  # Direct Input
                    age = st.number_input("Age", 10, 100, 30)
                    height = st.number_input("Height (cm)", 140, 200, 170)
                    weight = st.number_input("Weight (kg)", 40, 120, 70)
                
                # Calculate BMI regardless of input method
                bmi = round(weight / ((height / 100) ** 2), 2)
                
                # Display BMI
                st.metric("BMI", bmi, delta=None, delta_color="normal")
                
            # Column 2 inputs
            with col2:
                # Gender is a select box in both cases
                gender = st.selectbox("Gender", options=["Male", "Female"])
                
                if input_method == "Sliders":
                    duration = st.slider("Exercise Duration (min)", 0, 60, 20)
                    heart_rate = st.slider("Heart Rate (bpm)", 60, 180, 80)
                    body_temp = st.slider("Body Temperature (°C)", 35.0, 42.0, 37.0, 0.1)
                else:  # Direct Input
                    duration = st.number_input("Exercise Duration (min)", 0, 60, 20)
                    heart_rate = st.number_input("Heart Rate (bpm)", 60, 180, 80)
                    body_temp = st.number_input("Body Temperature (°C)", 35.0, 42.0, 37.0, 0.1)
            
            # Create user dataframe for prediction
            user_data = {
                "Age": [age],
                "BMI": [bmi],
                "Duration": [duration],
                "Heart_Rate": [heart_rate],
                "Body_Temp": [body_temp]
            }
            
            # Handle gender encoding
            if 'Gender_Male' in st.session_state['X_columns']:
                user_data["Gender_Male"] = [1 if gender == "Male" else 0]
            else:
                user_data["Gender_male"] = [1 if gender == "Male" else 0]
            
            # Ensure all columns match the training data
            user_df = pd.DataFrame(user_data)
            missing_cols = set(st.session_state['X_columns']) - set(user_df.columns)
            for col in missing_cols:
                user_df[col] = 0
                
            # Ensure column order matches training data
            user_df = user_df[st.session_state['X_columns']]
            
            # Display user parameter summary
            st.subheader("📊 Your Parameters Summary")
            display_df = pd.DataFrame({
                "Age": [age],
                "Height (cm)": [height],
                "Weight (kg)": [weight],
                "BMI": [bmi],
                "Duration (min)": [duration],
                "Heart Rate (bpm)": [heart_rate],
                "Body Temp (°C)": [body_temp],
                "Gender": [gender]
            })
            st.dataframe(display_df)
            
            # Model selection for prediction
            st.subheader("🔥 Calorie Prediction")
            
            available_models = list(st.session_state['best_models'].keys())
            selected_model = st.selectbox("Select Model for Prediction", 
                                        options=available_models,
                                        index=available_models.index(st.session_state['best_model_name']))
            
            model = st.session_state['best_models'][selected_model]
            prediction = model.predict(user_df)[0]
            
            # Display prediction with larger font and visually appealing
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"""
                <div style='padding: 20px; border-radius: 10px; background-color: #f0f2f6;'>
                <h2 style='text-align: center;'>🔥 {round(prediction, 2)} kcal 🔥</h2>
                <p style='text-align: center;'>Predicted with {selected_model}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Progress bar for visual representation
                max_calories = df["Calories"].max()
                progress = min(prediction / max_calories, 1.0)
                st.write("Relative to maximum in dataset:")
                st.progress(progress)
                
                # Add calorie burn context
                avg_calories = df["Calories"].mean()
                if prediction > avg_calories:
                    st.info(f"🔥 Your predicted burn is {round(prediction - avg_calories, 2)} kcal above average!")
                else:
                    st.info(f"📊 Your predicted burn is {round(avg_calories - prediction, 2)} kcal below average.")
            
            # Make predictions with all models for comparison
            if st.checkbox("Compare predictions from all models"):
                predictions = {}
                for model_name, model_obj in st.session_state['best_models'].items():
                    pred = model_obj.predict(user_df)[0]
                    predictions[model_name] = round(pred, 2)
                
                pred_df = pd.DataFrame({
                    'Model': list(predictions.keys()),
                    'Predicted Calories': list(predictions.values())
                })
                
                st.write("Predictions from all models:")
                st.write(pred_df)
                
                # Visualize model predictions
                fig = px.bar(pred_df, x='Model', y='Predicted Calories', 
                           title="Calorie Predictions Across Models", 
                           color='Model', text='Predicted Calories')
                fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            # Visualize User vs. Dataset
            st.subheader("📈 User vs. Dataset Comparison")
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.histogram(df, x="Age", nbins=20, title="Age Distribution", 
                                   color_discrete_sequence=["#FFA07A"])
                fig1.add_vline(x=age, line_color="red", line_dash="dash", 
                             annotation_text="Your Age")
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.histogram(df, x="BMI", nbins=20, title="BMI Distribution", 
                                   color_discrete_sequence=["#87CEFA"])
                fig2.add_vline(x=bmi, line_color="red", line_dash="dash", 
                             annotation_text="Your BMI")
                st.plotly_chart(fig2, use_container_width=True)
            
            # Similar Calories Burned
            calorie_range = [prediction - 10, prediction + 10]
            similar_data = df[(df["Calories"] >= calorie_range[0]) & (df["Calories"] <= calorie_range[1])]
            
            if not similar_data.empty:
                st.subheader("🧩 Similar Caloric Burn Profiles")
                st.write(similar_data.sample(min(5, len(similar_data))))
            
            # User Percentiles
            st.subheader("📈 Comparative Analysis")
            metrics = [
                {"label": "Age", "value": age, "column": "Age"},
                {"label": "BMI", "value": bmi, "column": "BMI"},
                {"label": "Exercise Duration", "value": duration, "column": "Duration"},
                {"label": "Heart Rate", "value": heart_rate, "column": "Heart_Rate"},
                {"label": "Body Temperature", "value": body_temp, "column": "Body_Temp"}
            ]
            
            for metric in metrics:
                percentile = round((df[metric["column"]] < metric["value"]).mean() * 100, 2)
                st.write(f"🔹 Your {metric['label']} is higher than {percentile}% of people in the dataset.")
            
            # Feature importance if Random Forest was used
            if "Random Forest Regressor" in st.session_state['best_models']:
                rf_model = st.session_state['best_models']["Random Forest Regressor"]
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.subheader("🎯 Feature Importance")
                fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                           title="Feature Importance in Predicting Calories Burned")
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.error("Please ensure your CSV files have the correct format. The calories file should have 'User_ID' and 'Calories' columns. The exercise file should have 'User_ID', 'Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', and 'Body_Temp' columns.")

else:
    # Instructions when files are not uploaded
    st.info("Please upload both the calories.csv and exercise.csv files to begin.")
    
    # Display sample file format
    st.subheader("Expected File Format")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**calories.csv**")
        calories_sample = pd.DataFrame({
            'User_ID': [1, 2, 3],
            'Calories': [240, 325, 180]
        })
        st.write(calories_sample)
    
    with col2:
        st.write("**exercise.csv**")
        exercise_sample = pd.DataFrame({
            'User_ID': [1, 2, 3],
            'Gender': ['male', 'female', 'male'],
            'Age': [25, 30, 35],
            'Height': [180, 165, 175],
            'Weight': [75, 62, 80],
            'Duration': [25, 30, 20],
            'Heart_Rate': [110, 125, 105],
            'Body_Temp': [37.5, 38.1, 37.3]
        })
        st.write(exercise_sample)