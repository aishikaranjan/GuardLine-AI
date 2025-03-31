import face_recognition as frg
import pickle as pkl
import os
import cv2
import numpy as np
import yaml
from collections import defaultdict
from PIL import Image

information = defaultdict(dict)

# Absolute path to the config.yaml directory
config_dir = os.path.dirname(r'C:/Users/aishi/OneDrive/Desktop/Capstone 3/Capstone 2/Face_recognition/config.yaml')

# Load config.yaml
cfg = yaml.load(open(os.path.join(config_dir, 'config.yaml'), 'r'), Loader=yaml.FullLoader)

# Absolute paths based on the config.yaml directory
DATASET_DIR = os.path.join(config_dir, cfg['PATH']['DATASET_DIR'])
PKL_PATH = os.path.join(config_dir, cfg['PATH']['PKL_PATH'])

def get_databse():
    with open(PKL_PATH, 'rb') as f:
        database = pkl.load(f)
    return database

def recognize(image, TOLERANCE):
    # Ensure image is in numpy array format
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    database = get_databse()
    known_encoding = [database[id]['encoding'] for id in database.keys()]
    name = 'Unknown'
    id = 'Unknown'
    
    # Detect face locations and encode them
    face_locations = frg.face_locations(image)
    face_encodings = frg.face_encodings(image, face_locations)
    
    # Iterate through faces and match them with known encodings
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = frg.compare_faces(known_encoding, face_encoding, tolerance=TOLERANCE)
        distance = frg.face_distance(known_encoding, face_encoding)
        
        # Default to unknown if no match is found
        name = 'Unknown'
        id = 'Unknown'
        
        # If a match is found, update the name and ID
        if True in matches:
            match_index = matches.index(True)
            name = database[match_index]['name']
            id = database[match_index]['id']
            distance = round(distance[match_index], 2)
            cv2.putText(image, str(distance), (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    
    return image, name, id

def isFaceExists(image):
    face_location = frg.face_locations(image)
    return len(face_location) > 0

def submitNew(name, id, image, old_idx=None):
    database = get_databse()
    
    # If the image isn't a NumPy array, convert it
    if type(image) != np.ndarray:
        image = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)

    # Check if face exists in image
    if not isFaceExists(image):
        return -1  # No face detected
    
    # Encode image
    encodings = frg.face_encodings(image)
    if not encodings:
        return -1  # Return error if no face encoding found
    encoding = encodings[0]

    # Append to database
    existing_id = [database[i]['id'] for i in database.keys()]
    
    # Update mode
    if old_idx is not None:
        new_idx = old_idx
    else:
        # Add mode: Check if ID already exists
        if id in existing_id:
            return 0  # ID already exists
        new_idx = len(database)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    database[new_idx] = {'image': image,
                         'id': id,
                         'name': name,
                         'encoding': encoding}
    
    # Save database to file
    with open(PKL_PATH, 'wb') as f:
        pkl.dump(database, f)
    
    return True

def get_info_from_id(id):
    database = get_databse()
    for idx, person in database.items():
        if person['id'] == id:
            name = person['name']
            image = person['image']
            return name, image, idx
    return None, None, None

def deleteOne(id):
    database = get_databse()
    id = str(id)
    for key, person in database.items():
        if person['id'] == id:
            del database[key]
            break
    with open(PKL_PATH, 'wb') as f:
        pkl.dump(database, f)
    return True

def build_dataset():
    counter = 0
    for image in os.listdir(DATASET_DIR):
        image_path = os.path.join(DATASET_DIR, image)
        image_name = image.split('.')[0]
        parsed_name = image_name.split('_')
        person_id = parsed_name[0]
        person_name = ' '.join(parsed_name[1:])
        
        # Ensure only JPG images are processed
        if not image_path.endswith('.jpg'):
            continue
        
        image = frg.load_image_file(image_path)
        information[counter]['image'] = image
        information[counter]['id'] = person_id
        information[counter]['name'] = person_name
        information[counter]['encoding'] = frg.face_encodings(image)[0]
        counter += 1

    # Save dataset to pickle file
    with open(os.path.join(DATASET_DIR, 'database.pkl'), 'wb') as f:
        pkl.dump(information, f)

if __name__ == "__main__":
    deleteOne(4)
