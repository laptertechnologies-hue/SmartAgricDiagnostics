import tensorflow_datasets as tfds
import os
import tensorflow as tf

def prepare_plantvillage():
    print("Downloading PlantVillage dataset (this may take a while)...")
    # Load PlantVillage dataset
    dataset, info = tfds.load('plant_village', split='train', with_info=True)
    
    # Define target directory
    TARGET_DIR = "data/train"
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    class_names = info.features['label'].names
    for class_name in class_names:
        os.makedirs(os.path.join(TARGET_DIR, class_name), exist_ok=True)
        
    print(f"Dataset downloaded. Found {len(class_names)} classes.")
    
    print("Exporting images to local directory structure...")
    count = 0
    for example in dataset:
        image = example['image'].numpy()
        label = example['label'].numpy()
        class_name = class_names[label]
        
        # Filter to only keep Maize (Corn) for this prototype, as PlantVillage 
        # doesn't contain Cassava or Beans by default.
        if 'Corn' in class_name:
            img_path = os.path.join(TARGET_DIR, class_name, f"img_{count}.jpg")
            tf.keras.utils.save_img(img_path, image)
            count += 1
            if count % 500 == 0:
                print(f"Exported {count} images...")
                
    print(f"Completed! Exported {count} relevant Corn (Maize) images to {TARGET_DIR}.")
    print("Note: For Cassava and Beans, you will need to download custom datasets (e.g., Makerere AI Lab or iCassava 2019) and place them in the 'data/train' directory.")

if __name__ == '__main__':
    prepare_plantvillage()
