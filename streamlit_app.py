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

# Add Icons and Reorder Items
page = st.sidebar.selectbox(
    'Go to', 
    ['🏠 Home', '🔍 Nutrition Prediction', '📊 Nutrition Data Table', '📈 Visualizations Data Kalori', '🖼️ Visualizations Gambar Makanan', 'ℹ️ About Us']
)

# Function to inject custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Add Separators
st.sidebar.markdown("---")

# Convert selected page to lowercase for URL
selected_page = page.lower().replace(' ', '_')

if selected_page == '🏠_home':
    st.header("🍲 Aplikasi Prediksi Gambar Makanan 📸")
    st.markdown("""
    Selamat datang di Aplikasi Prediksi Gambar Makanan. Aplikasi ini dirancang untuk 
    memprediksi makanan dari gambar yang diunggah dan memberikan informasi nutrisi tentang 
    makanan tersebut, termasuk kalori. Selain itu, Anda juga dapat melihat perkiraan jarak yang 
    perlu Anda tempuh untuk membakar kalori tersebut. 🏃‍♂️💪
""")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.header("🎯 Akurasi Model")
        if selected_model == "keras_model_best.h5":
            st.success("Model terbaik ini memiliki akurasi sebesar 96% dengan dataset gambar yang terdiri dari 10 makanan. 👍")
        else:
            st.warning("Model ini memiliki akurasi sebesar 86% dengan dataset gambar yang terdiri dari 30 makanan. ⚠️")

    with col2:
        st.header("📋 Daftar Makanan")
        st.write("Berikut adalah daftar makanan yang dapat diprediksi menggunakan model-model ini:")
        data = {'Makanan': menu}
        df = pd.DataFrame(data)
        st.dataframe(df, width=500, height=200)

    st.markdown("---")

    if selected_model == "keras_model_best.h5":
        st.subheader("🚀 Visualisasi Model Terbaik")
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
        st.subheader("🚀 Visualisasi Model")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image("Accuracy.png", caption="Akurasi Model 1", width=300)
            st.image("Accuracy_Epoch.png", caption="Akurasi Epoch", width=300)
        st.markdown("---")
        with col2:
            st.image("Accuracy2.png", caption="Akurasi Model 2", width=300)
            st.image("Loss.jpg", caption="Loss Model", width=300)
        st.image("Confusion.png", caption="Confusion Matrix", use_column_width=True)

elif selected_page == '🔍_nutrition_prediction':
    st.title("🥗 Nutrition Prediction from Image")

    uploaded_file = st.file_uploader("Choose an image... 🖼️", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with st.spinner('Analyzing the image... Please wait... 🔄'):
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
                    st.subheader('Nutrition Information 📊')
                    st.json(nutrition_values)  # Using st.json to give it a card-like UI

                    regression_features = np.array([[nutrition_info['calories']]])
                    regression_prediction = regression_model.predict(regression_features)

        # Display results in a more styled way
        st.markdown(f"""
            <h2 style="text-align: center;">{predicted_class_name.split(' ', 1)[1]}</h2>
            <h3 style="text-align: center;">Confidence: {confidence_score:.2%} 🎯</h3>
            <div style="display: flex; justify-content: space-between;">
                <p><b>Predicted Distance:</b> {regression_prediction[0][1]:.1f} meters 🏃‍♂️</p>
                <p style="text-align: right;"><b>Predicted Steps:</b> {regression_prediction[0][0]:.1f} steps 👟</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No image uploaded yet. Please upload an image to predict its nutritional content. 🆙")

elif selected_page == '📊_nutrition_data_table':
    st.title("📈 Nutrition Data Overview")

    # Clear and concise instructions or descriptions
    st.markdown("Explore the nutritional values of various foods. Use the search box to filter by food name. 🍲")

    # Drop unwanted columns in a cleaner way
    columns_to_drop = ['id', 'image']
    nutrition_data = nutrition_data.drop(columns=[col for col in columns_to_drop if col in nutrition_data.columns])

    # Make sure we only have rows with non-zero values for the specified columns
    nutrition_data = nutrition_data[(nutrition_data[['calories', 'proteins', 'fat', 'carbohydrate']] != 0).all(axis=1)]

    # Calculate protein to calorie ratio
    nutrition_data = calculate_protein_calorie_ratio(nutrition_data)

    # Search functionality with a friendly placeholder
    search_query = st.text_input("🔍 Search for a food", placeholder="Enter a food name to search")

    # Sort the data based on the protein to calorie ratio or any other user-selected criteria
    sort_column = st.selectbox("Sort by", options=['protein_to_calorie_ratio', 'calories', 'proteins', 'fat', 'carbohydrate'], index=0)
    ascending_order = st.checkbox("Ascending order", value=False)

    if search_query:
        nutrition_data = nutrition_data[nutrition_data['name'].str.contains(search_query, case=False, na=False)]
    
    nutrition_data = nutrition_data.sort_values(by=sort_column, ascending=ascending_order)
    formatted_nutrition_data = format_data(nutrition_data)

    # Ensure the dataframe columns are numeric before applying numeric formatting
    def format_dataframe(df):
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].apply(lambda x: f'{x:.2f}')
            else:
                df[col] = df[col].apply(lambda x: f'{x}')  # Keep string columns unchanged
        return df

    styled_data = format_dataframe(formatted_nutrition_data)

    # Create an interactive table with the sorted and formatted data
    st.markdown("### Nutritional Data Table")
    st.dataframe(styled_data, height=400)

    # Additional tip or note for the user
    st.markdown("*Note: You can sort the data by clicking on the column headers.*")

elif selected_page == '📈_visualizations_data_kalori':
    st.title("📊 Activity Data Over Time")
    st.markdown("""
        This section explores the relationships between different activity metrics over time.
        Use the date pickers below to select the time period you're interested in. 📅
    """)

    # Interactive date pickers for user selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Start Date', value=pd.to_datetime('2022-08-03'))
    with col2:
        end_date = st.date_input('End Date', value=pd.to_datetime('2023-05-14'))
    
    # Load and preprocess data
    calorie_data = pd.read_csv('Data.csv')
    # Convert the 'date' column to datetime if it's not already
    calorie_data['date'] = pd.to_datetime(calorie_data['date'])
    cleaned_data = clean_data(calorie_data)
    # Ensure that the comparison is between datetime objects
    filtered_data = cleaned_data[(cleaned_data['date'] >= pd.to_datetime(start_date)) & (cleaned_data['date'] <= pd.to_datetime(end_date))]

    # Summary statistics with a toggle
    if st.checkbox('Show Summary Statistics'):
        st.dataframe(filtered_data.describe())

    # Lineplot for steps, calories, and distance with improved layout
    st.subheader("📈 Metrics Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='date', y='steps', data=filtered_data, ax=ax, label='Steps 🚶', color='blue')
    sns.lineplot(x='date', y='calories', data=filtered_data, ax=ax, label='Calories 🔥', color='red')
    sns.lineplot(x='date', y='distance', data=filtered_data, ax=ax, label='Distance 📏', color='green')
    ax.set_title('Activity Data Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Metrics')
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(DayLocator(interval=7))
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Distribution plots with better alignment and captions
    st.subheader("📊 Distribution of Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.image("Distribution of Calories.png", caption="Calories Distribution 🔥", use_column_width=True)
        st.image("Distribution of Distance.png", caption="Distance Distribution 📏", use_column_width=True)
    with col2:
        st.image("Distribution of runDistance.png", caption="Run Distance Distribution 🏃", use_column_width=True)
        st.image("Distribution Steps.png", caption="Steps Distribution 🚶", use_column_width=True)

    # Additional data visualizations with contextual captions
    st.subheader("📈 Additional Data Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.image("Data.png", caption="Various Data Points 📍", use_column_width=True)
    with col2:
        st.image("Heatmap Data.png", caption="Metrics Correlation Heatmap 🔥", use_column_width=True)

elif selected_page == '🖼️_visualizations_gambar_makanan':
    st.title("🍽️ Gallery of Food Images")
    st.markdown("Browse the gallery below to explore a variety of food images. 🖼️🥘")

    directory = './Merged_Dataset' if selected_model == "keras_model.h5" else './Merged_Dataset_Best'
    subfolders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    if subfolders:
        folder_choice = st.selectbox("Choose a category to view images from:", subfolders)
        folder_path = os.path.join(directory, folder_choice)
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
        
        if image_files:
            # Sort the image files so that they are displayed in order
            image_files.sort()
            # Create a container to make it scrollable
            with st.container():
                # Iterate over images in steps of 5 to create rows
                for i in range(0, len(image_files), 5):
                    # Create a row of 5 columns
                    cols = st.columns(5)
                    # For each column, display an image and its filename
                    for col, image_path in zip(cols, image_files[i:i+5]):
                        with col:
                            st.image(image_path, use_column_width=True)
                            st.caption(os.path.basename(image_path))
        else:
            st.error("No image files found in the selected folder.")
    else:
        st.error("No subfolders found in the dataset directory.")
    
elif selected_page == 'ℹ️_about_us':
    st.title("Dataset Information 📊")

    st.markdown("""
        ### 🍜 Dataset Makanan
        Here are some datasets related to Indonesian cuisine that have been utilized:

        1. [Kue Indonesia](https://www.kaggle.com/datasets/ilhamfp31/kue-indonesia) - A dataset featuring Indonesian cakes.
        2. [Makanan Indonesia](https://www.kaggle.com/datasets/theresalusiana/indonesian-food) - General dataset of Indonesian food.
        3. [Kue Tradisional Indonesia](https://www.kaggle.com/datasets/widyaameliaputri/indonesian-traditional-cakes) - Traditional Indonesian cakes.
        4. [Makanan Padang](https://www.kaggle.com/datasets/faldoae/padangfood) - Foods specific to the Padang region.
        5. [Jajanan Tradisional Jawa Tengah](https://www.kaggle.com/datasets/nizamkurniawan/jajanan-tradisional-jawa-tengah) - Traditional snacks from Central Java.
    """)

    st.markdown("""
        ### 📝 Dataset Nutrisi
        Nutrition information is crucial for understanding the health impact of different foods. The dataset used is the [Indonesian Food and Drink Nutrition Dataset](https://www.kaggle.com/datasets/anasfikrihanif/indonesian-food-and-drink-nutrition-dataset), which provides comprehensive nutritional details.
    """)

    st.markdown("""
        ### 🏋️‍♂️ Dataset Kalori
        The calorie dataset is a personal dataset obtained from the export of a fitness tracking application used by Taruna Rakha Maulana, providing insights into calories burned during various activities.
    """)
