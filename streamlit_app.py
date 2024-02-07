from matplotlib.dates import DateFormatter, DayLocator
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
import joblib
import os
from streamlit.components.v1 import html

# Create a pop-up to select the model and dataset
st.sidebar.title("Select Model and Dataset")
selected_model = st.sidebar.selectbox("Select Model", ["keras_model.h5", "keras_model_best.h5"])
selected_dataset = "Merged_Dataset" if selected_model == "keras_model.h5" else "Merged_Dataset_Best"
regression_model = joblib.load('Calories_Detector.h5')
nutrition_data = pd.read_csv("nutrition.csv")
label_file = "labels.txt" if selected_model == "keras_model.h5" else "labels_best.txt"

# Load the selected model
if selected_model:
    classification_model = load_model(selected_model, compile=False)
    class_names = [line.strip() for line in open(label_file, "r").readlines()]

def clean_data(data):
    data = data.dropna()
    numeric_cols = ['steps', 'distance', 'runDistance', 'calories']
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    Q1 = data[numeric_cols].quantile(0.25)
    Q3 = data[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data[numeric_cols] < (Q1 - 1.5 * IQR)) | (data[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
    data[numeric_cols] = data[numeric_cols].astype(float)
    return data

def plot_calorie_relationships(data):
    plt.figure(figsize=(15, 5))
    features = ['steps', 'distance', 'runDistance']
    for i, feature in enumerate(features, 1):
        plt.subplot(1, 3, i)
        sns.scatterplot(x=data[feature], y=data['calories'])
        plt.title(f'Calories vs {feature.capitalize()}')
        plt.xlabel(feature.capitalize())
        plt.ylabel('Calories')
    plt.tight_layout()
    return plt

def load_data():
    data = pd.read_csv('data.csv')
    return data

def format_data(data_frame):
    for col in ['calories', 'proteins', 'fat', 'carbohydrate']:
        data_frame[col] = data_frame[col].astype(float).map('{:.1f}'.format)
    return data_frame

def calculate_protein_calorie_ratio(data_frame):
    data_frame = data_frame[data_frame['calories'] > 0]
    data_frame['protein_to_calorie_ratio'] = data_frame['proteins'] / data_frame['calories']
    return data_frame

def sort_nutrition_data(data_frame, column, ascending=True):
    if column == 'calorie_to_protein_ratio':
        if 'calorie_to_protein_ratio' not in data_frame.columns:
            data_frame['calorie_to_protein_ratio'] = data_frame['calories'] / data_frame['proteins']
        sorted_frame = data_frame.sort_values(by='calorie_to_protein_ratio', ascending=ascending)
    else:
        sorted_frame = data_frame.sort_values(by=column, ascending=ascending)
    return format_data(sorted_frame)

def get_predictions(image_array):
    predictions = classification_model.predict(np.array([image_array]))
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class

with open(label_file, "r") as label_file:
    labels = label_file.readlines()

menu = [label.strip() for label in labels]

# UI/UX improvements
st.sidebar.title("Navigation")
page = st.sidebar.selectbox('Go to', ['Home','Nutrition Prediction', 'Nutrition Data Table', 'Visualizations Data Kalori', 'Visualizations Gambar Makanan','About US'])

# Function to inject custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

if page == 'Home':
    st.header("Aplikasi Prediksi Gambar Makanan")
    st.markdown("""
    Selamat datang di Aplikasi Prediksi Gambar Makanan. Aplikasi ini dirancang untuk 
    memprediksi makanan dari gambar yang diunggah dan memberikan informasi nutrisi tentang 
    makanan tersebut, termasuk kalori. Selain itu, Anda juga dapat melihat perkiraan jarak yang 
    perlu Anda tempuh untuk membakar kalori tersebut.
""")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.header("Akurasi Model")
        if selected_model == "keras_model_best.h5":
            st.success("Model terbaik ini memiliki akurasi sebesar 96% dengan dataset gambar yang terdiri dari 10 makanan.")
        else:
            st.warning("Model ini memiliki akurasi sebesar 86% dengan dataset gambar yang terdiri dari 30 makanan.")

    with col2:
        st.header("Daftar Makanan")
        st.write("Berikut adalah daftar makanan yang dapat diprediksi menggunakan model-model ini:")
        data = {'Makanan': menu}
        df = pd.DataFrame(data)
        st.dataframe(df, width=500, height=200)

    st.markdown("---")

    if selected_model == "keras_model_best.h5":
        st.subheader("Visualisasi Model Terbaik")
        st.image("Accuracy_Best.jpg", caption="Akurasi Model Terbaik", width=300)
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(["Accuracy_Epoch_Best.png"], caption=["Akurasi Epoch"], width=300)
            st.markdown("---")
        with col2:
            st.image(["Loss_Best.jpg"], caption=["Loss Model"], width=300)
        st.image("Confusion_Best.jpg", caption="Confusion Matrix", use_column_width=True)
    else:
        st.subheader("Visualisasi Model")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image("Accuracy.png", caption="Akurasi Model 1", width=300)
            st.image("Accuracy_Epoch.png", caption="Akurasi Epoch", width=300)
        st.markdown("---")
        with col2:
            st.image("Accuracy2.png", caption="Akurasi Model 2", width=300)
            st.image("Loss.jpg", caption="Loss Model", width=300)
        st.image("Confusion.png", caption="Confusion Matrix", use_column_width=True)

elif page == 'Nutrition Prediction':
    st.title("Nutrition Prediction from Image")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        prediction = classification_model.predict(data)
        index = np.argmax(prediction)
        predicted_class_name = class_names[index].strip()
        confidence_score = prediction[0][index]
        
        col1, col2 = st.columns([1.5, 2.5])
        
        with col1:
            st.image(image, caption="Uploaded Image.", use_column_width=True)
        
        predicted_food_name = predicted_class_name.split(" ", 1)[1]  # Remove numeric ID
        nutrition_info = nutrition_data[nutrition_data['name'].str.contains(predicted_food_name, case=False, na=False)]
        
        if not nutrition_info.empty:
            nutrition_info = nutrition_info.iloc[0]  # Use the first match
            nutrition_values = {
                'Calories': round(nutrition_info['calories'], 1),
                'Proteins': round(nutrition_info['proteins'], 1),
                'Fat': round(nutrition_info['fat'], 1),
                'Carbohydrates': round(nutrition_info['carbohydrate'], 1)
            }
            
            with col2:
                st.subheader('Nutrition Information')
                st.table(pd.DataFrame.from_dict(nutrition_values, orient='index', columns=['Amount']))
                
                regression_features = np.array([[nutrition_info['calories']]])
                regression_prediction = regression_model.predict(regression_features)
        
        st.markdown("""
            <style>
            .prediction-header {
                text-align: center;
                font-size: 24px;
                font-weight: bold;
            }
            .confidence-score {
                text-align: center;
            }
            .nutrition-info {
                display: flex;
                justify-content: space-between;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.write(f"<p class='prediction-header'>{predicted_class_name.split(' ', 1)[1]}</p>", unsafe_allow_html=True)
        st.write(f"<p class='confidence-score'>Confidence Score: {confidence_score:.4f}</p>", unsafe_allow_html=True)
        st.write(f"<div class='nutrition-info'>Predicted Distance: {regression_prediction[0][1]:.1f} meter  <div style='text-align: right;'>Predicted Steps: {regression_prediction[0][0]:.1f}</div></div>", unsafe_allow_html=True)
            
elif page == 'Nutrition Data Table':
    st.title("Nutrition Data Table")
    if 'id' in nutrition_data.columns:
        nutrition_data = nutrition_data.drop(['id'], axis=1)
    if 'image' in nutrition_data.columns:
        nutrition_data = nutrition_data.drop(['image'], axis=1)
    nutrition_data = nutrition_data[(nutrition_data[['calories', 'proteins', 'fat', 'carbohydrate']] != 0).all(axis=1)]
    
    st.write('<style>div.Widget.row-widget.stDataFrame{overflow: auto !important;}</style>', unsafe_allow_html=True)
    
    st.dataframe(nutrition_data.describe(), width=750)  # Sesuaikan tinggi sesuai kebutuhan
    
    nutrition_data = nutrition_data[(nutrition_data[['calories', 'proteins', 'fat', 'carbohydrate']] != 0).all(axis=1)]
    nutrition_data = calculate_protein_calorie_ratio(nutrition_data)
    search_query = st.text_input("Enter a food name to search:")
    if search_query:
        nutrition_data = nutrition_data[nutrition_data['name'].str.contains(search_query, case=False, na=False)]
    nutrition_data = nutrition_data.sort_values(by='protein_to_calorie_ratio', ascending=False)
    nutrition_data = format_data(nutrition_data)
    
    st.write('<style>div.Widget.row-widget.stDataFrame{overflow: auto !important;}</style>', unsafe_allow_html=True)
    
    st.dataframe(nutrition_data)

    
elif page == 'Visualizations Data Kalori':
    st.title("Activity Data Over Time")
    st.write("This section displays the relationships in Data.csv over a specified time period.")
    
    # Load and preprocess data
    calorie_data = pd.read_csv('Data.csv')
    start_date = '2022-08-03'
    end_date = '2023-05-14'
    cleaned_data = clean_data(calorie_data)
    filtered_data = cleaned_data[(cleaned_data['date'] >= start_date) & (cleaned_data['date'] <= end_date)]
    filtered_data['date'] = pd.to_datetime(filtered_data['date'])
    
    # Show statistics of the data
    st.write("### Summary Statistics:")
    st.dataframe(filtered_data.describe(), height=300)
    
    # Create lineplot for steps, calories, and distance
    st.subheader("Metrics Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='date', y='steps', data=filtered_data, ax=ax, label='Steps', color='blue')
    sns.lineplot(x='date', y='calories', data=filtered_data, ax=ax, label='Calories', color='red')
    sns.lineplot(x='date', y='distance', data=filtered_data, ax=ax, label='Distance', color='green')
    ax.set_title('Activity Data Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Metrics')
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(DayLocator(interval=7))
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show distribution plots
    st.subheader("Distribution of Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.image("Distribution of Calories.png", caption="Distribution of Calories", use_column_width=True)
        st.image("Distribution of Distance.png", caption="Distribution of Distance", use_column_width=True)
    with col2:
        st.image("Distribution of runDistance.png", caption="Distribution of Run Distance", use_column_width=True)
        st.image("Distribution Steps.png", caption="Distribution of Steps", use_column_width=True)
    
    # Show additional data visualizations
    st.subheader("Additional Data Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.image("Data.png", caption="Data", use_column_width=True)
    with col2:
        st.image("Heatmap Data.png", caption="Heatmap Data", use_column_width=True)

elif page == 'Visualizations Gambar Makanan':
    st.title("Contoh Gambar Makanan")
    st.write("Berikut adalah contoh gambar makanan dari berbagai folder.")

    # Tentukan direktori dan model yang akan digunakan
    directory = './Merged_Dataset' if selected_model == "keras_model.h5" else './Merged_Dataset_Best'
    subfolders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    num_subplots = len(subfolders)
    num_cols = 5 
    num_rows = (num_subplots + num_cols - 1) // num_cols  # Number of rows in the subplot grid
    
    # Buat grid untuk menampilkan gambar-gambar
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, 16), constrained_layout=True)
    
    for i, folder in enumerate(subfolders):
        folder_path = os.path.join(directory, folder)
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
        if len(image_files) > 0:
            image_path = os.path.join(folder_path, image_files[0])
            image = Image.open(image_path)
            row_idx = i // num_cols
            col_idx = i % num_cols
            axes[row_idx, col_idx].imshow(image)
            axes[row_idx, col_idx].set_title(folder, fontsize=12, pad=5)  # Penambahan padding dan ukuran font
            axes[row_idx, col_idx].axis('off')
            axes[row_idx, col_idx].set_aspect('equal')
    
    if num_subplots < num_rows * num_cols:
        for i in range(num_subplots, num_rows * num_cols):
            row_idx = i // num_cols
            col_idx = i % num_cols
            fig.delaxes(axes[row_idx, col_idx])
    
    # Tambahkan jarak antara gambar dan tulisan
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    # Tampilkan plot menggunakan st.pyplot()
    st.pyplot(fig)
    
    # Tambahkan garis horizontal sebagai pemisah
    st.markdown("***")
    
elif page == 'About US':
    st.title("Dataset Information 📊")

    # Informasi mengenai dataset makanan
    st.header("Dataset Makanan")
    st.write("1. [Kue Indonesia](https://www.kaggle.com/datasets/ilhamfp31/kue-indonesia)")
    st.write("2. [Makanan Indonesia](https://www.kaggle.com/datasets/theresalusiana/indonesian-food)")
    st.write("3. [Kue Tradisional Indonesia](https://www.kaggle.com/datasets/widyaameliaputri/indonesian-traditional-cakes)")
    st.write("4. [Makanan Padang](https://www.kaggle.com/datasets/faldoae/padangfood)")
    st.write("5. [Jajanan Tradisional Jawa Tengah](https://www.kaggle.com/datasets/nizamkurniawan/jajanan-tradisional-jawa-tengah)")

    # Informasi mengenai dataset nutrisi
    st.header("Dataset Nutrisi")
    st.write("Dataset Nutrisi: [Indonesian Food and Drink Nutrition Dataset](https://www.kaggle.com/datasets/anasfikrihanif/indonesian-food-and-drink-nutrition-dataset)")

    # Informasi mengenai dataset kalori
    st.header("Dataset Kalori")
    st.write("Dataset Kalori merupakan data pribadi yang didapat dari export aplikasi pelacak kebugaran yang digunakan oleh Taruna Rakha Maulana.")