import os
import pandas as pd
import xml.etree.ElementTree as ET
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def run():

    # Membuat Title
    st.title('Road Signs Detections')

    # Membuat subheader
    st.subheader('EDA for Road signs detection')

    # Menampilkan pembuat page
    st.write('Made by Fahri')

    # Bold dan italic contoh teks
    st.write('**Road Signs Data Analysis**')
    st.write('*using road signs dataset with 4 classes*')

    # Membuat garis lurus
    st.markdown('---')
    
    # Load images and labels
    data_dir = r'C:\Users\C O R E I 5\Documents\Hacktiv8\fase 2\road-signs-data'
    image_paths = []
    labels = []

    # Walk through the directory to collect image paths and labels
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".xml"):
                image_paths.append(os.path.join(root, file))
                labels.append(os.path.basename(root))

    # Paths for annotations and images
    main_path = 'road-signs-data'
    annotations_path = os.path.join(main_path, 'annotations')
    images_path = os.path.join(main_path, 'images')

    # Function to parse XML annotations
    def parse_annotation(xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        boxes = []
        label = None

        for obj in root.iter('object'):
            label = obj.find('name').text
            for bbox in obj.findall('bndbox'):
                box = [
                    int(bbox.find('xmin').text),
                    int(bbox.find('ymin').text),
                    int(bbox.find('xmax').text),
                    int(bbox.find('ymax').text)
                ]
                boxes.append(box)

        return label, boxes

    # Process annotations to create DataFrame
    data = []
    for xml_file in os.listdir(annotations_path):
        if xml_file.endswith('.xml'):
            # Derive image name from XML filename
            image_name = os.path.splitext(xml_file)[0] + '.png'
            label, boxes = parse_annotation(os.path.join(annotations_path, xml_file))

            # Check if image exists
            image_path = os.path.join(images_path, image_name)
            if os.path.exists(image_path):
                for box in boxes:
                    data.append({'Image_Name': image_name, 'Label': label, 'Bounding Box': box})
            else:
                st.write(f"Image not found: {image_path}")  # Display missing images message in Streamlit

    # Create DataFrame
    df = pd.DataFrame(data)
    # Convert to DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'label': labels
    })

    # Display data in Streamlit
    st.title("Road Sign Detection Data")
    st.write(f"Total images: {len(df)}")

    # Calculate class distribution
    class_counts = df['Label'].value_counts()
    classes = class_counts.index

    # Streamlit section for Class Distribution Bar Plot
    st.title("Class Distribution of Road Signs")

    # Bar plot for class distribution
    st.write("### Class Distribution")
    fig_bar, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=classes, y=class_counts, ax=ax)
    ax.set_title('Class Distribution of Road Signs')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Number of Images')
    st.pyplot(fig_bar)

    # Pie plot for class proportion
    st.write("### Proportion of Each Class in the Dataset")
    fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
    ax_pie.pie(class_counts, labels=classes, autopct='%1.1f%%', startangle=90, colors=['blue', 'red', 'green', 'orange'])
    ax_pie.set_title('Proportion of Each Class in the Dataset')
    ax_pie.axis('equal')
    st.pyplot(fig_pie)

    # Display one image from each class
    st.write("### Sample Images from Each Class")
    fig_img, axes = plt.subplots(1, 4, figsize=(20, 10))

    for i, class_name in enumerate(classes):
        # Make sure the correct path to the image folder is set
        image_path = os.path.join(images_path, class_name, os.listdir(os.path.join(images_path, class_name))[0])
        
        if os.path.exists(image_path):
            img = Image.open(image_path)
            axes[i].imshow(img)
            axes[i].set_title(class_name)
            axes[i].axis('off')

    st.pyplot(fig_img)
        
    # Run the Streamlit app
    if __name__ == "__main__":
        run()