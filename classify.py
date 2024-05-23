from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

loaded_model = joblib.load('iris_classifier.pkl')

iris_data = pd.read_csv('./data/iris.csv')
label_encoder = LabelEncoder()
label_encoder.fit(iris_data['Species'])

def extract_features_from_image(image_path):
    with Image.open(image_path) as img:
        original_img = img.copy()
        gray_img = img.convert('L')
        resized_img = gray_img.resize((150, 150))
        flattened_image = np.array(resized_img).flatten()
        features = flattened_image[:4]
        normalized_features = features / 255.0
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(gray_img, cmap='gray')
    axes[1].set_title("Grayscale Image")
    axes[1].axis('off')
    
    axes[2].imshow(resized_img, cmap='gray')
    axes[2].set_title("Resized Image (150x150)")
    axes[2].axis('off')
    
    plt.suptitle('Image Preprocessing Steps')
    plt.show()
    
    return normalized_features

def classify_image(image_path):
    image_features = extract_features_from_image(image_path)
    predicted_label = loaded_model.predict([image_features])[0]
    predicted_species = label_encoder.inverse_transform([predicted_label])[0]
    return predicted_species

if __name__ == "__main__":
    predicted_species = classify_image("./f1.png")
    print("Predicted species:", predicted_species)
    
    feature_importances = loaded_model.feature_importances_
    features = ['Pixel1', 'Pixel2', 'Pixel3', 'Pixel4']
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=features)
    plt.title('Feature Importance in Image Classifier')
    plt.show()
    
    y_true = label_encoder.transform(iris_data['Species'])
    y_pred = loaded_model.predict(iris_data.drop(columns=['Id', 'Species']).values[:, :4] / 255.0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
