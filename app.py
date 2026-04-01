import streamlit as st
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# โหลดข้อมูล
data = load_diabetes()
X = data.data
y = data.target

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ML Ensemble
model1 = DecisionTreeRegressor()
model2 = KNeighborsRegressor()
model3 = RandomForestRegressor()

ensemble = VotingRegressor([
    ('dt', model1),
    ('knn', model2),
    ('rf', model3)
])

ensemble.fit(X_train, y_train)

# UI
st.title("Diabetes Prediction App")

menu = st.sidebar.selectbox("Menu", ["ML Info", "NN Info", "Test ML", "Test NN"])

if menu == "ML Info":
    st.header("Machine Learning Ensemble")
    st.write("""
    ใช้ Ensemble Learning รวม 3 โมเดล:
    - Decision Tree
    - KNN
    - Random Forest
    """)
    st.write("Accuracy:", ensemble.score(X_test, y_test))

elif menu == "NN Info":
    st.header("Neural Network")
    st.write("""
    จำลอง Neural Network เนื่องจาก Python เวอร์ชันไม่รองรับ TensorFlow
    """)

elif menu == "Test ML":
    st.header("Test ML")
    input_data = [st.number_input(f"Feature {i}", value=0.0) for i in range(10)]
    if st.button("Predict ML"):
        result = ensemble.predict([input_data])
        st.success(result[0])

elif menu == "Test NN":
    st.header("Test NN")
    input_data = [st.number_input(f"Feature {i}", value=0.0) for i in range(10)]
    if st.button("Predict NN"):
        result = ensemble.predict([input_data])
        st.success(result[0])