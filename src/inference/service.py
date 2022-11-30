# For Mac computers - solve redirects problem, please comment the bottom 2 lines if you are a Windows user
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
################
from src.train.model import KWClassifier
#############
# Build flask server here that wraps our model instance
#############

# Object declaration
classifier = KWClassifier(h1_labels_mapping_path='../../data/level_1_mapping.csv',
                          h2_labels_mapping_path='../../data/level_2_mapping.csv',
                          h3_labels_mapping_path='../../data/level_3_mapping.csv',
                          h1_to_h2_mapping_path='../../data/h1_h2_mapping_indices.csv',
                          h2_to_h3_mapping_path='../../data/h2_h3_mapping_indices.csv',
                          taxonomy_path='../../data/new_intent_taxonomy_with_id.csv',
                          temperature_h1_path='../../data/temperature_h1.pkl',
                          temperature_h2_path='../../data/temperature_h2.pkl',
                          temperature_h3_path='../../data/temperature_h3.pkl',
                          is_negative_class=True)

# Start your service here
