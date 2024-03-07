import csv
import json

# Load CSV file
melanoma_file_path = '/Users/JUAN/Desktop/medical_vqa_vlm/data/processed/df_melanoma_clean.csv'
fusion_file_path = '/Users/JUAN/Desktop/medical_vqa_vlm/data/processed/df_fusion_clean.csv'

# JSON outputs
melanoma_output_path = '/Users/JUAN/Desktop/medical_vqa_vlm/data/processed//df_melanoma_qa2.json'
fusion_output_path = '/Users/JUAN/Desktop/medical_vqa_vlm/data/processed//df_fusion_qa2.json'

# Define initial question/answer format for each variable
# qa_pairs = {
#     'General Silhouette': {
#         'question': 'Can you describe the general silhouette?',
#         'answer_template': 'The general silhouette of the lesion is {value}.'
#     },
#     'Histology Diagnostic': {
#         'question': 'What is the histology diagnostic?',
#         'answer_template': 'The histology diagnostic indicates {value}.'
#     },
#     'Dysplasia': {
#         'question': 'Can you see any dysplasia?',
#         'answer_template': 'The presence of dysplasia is {value}.'
#     },
#     'Diagnostic Impression': {
#         'question': 'What is the diagnostic impression?',
#         'answer_template': 'The diagnostic impression is {value}.'
#     },
#     'Diagnostic Difficulty': {
#         'question': 'What is the diagnostic difficulty?',
#         'answer_template': 'The diagnostic difficulty level is {value}.'
#     },
#     'Excision': {
#         'question': 'Is there any excision?',
#         'answer_template': 'The lesion has been {value}.'
#     },
#     'Global Pattern': {
#         'question': 'How is the global pattern?',
#         'answer_template': 'The global pattern of the lesion is {value}.'
#     },
#     'Peripheral Globular Pattern': {
#         'question': 'How is the peripheral globular pattern?',
#         'answer_template': 'The peripheral globular pattern is {value}.'
#     },
#     'Symmetry': {
#         'question': 'Is there any symmetry pattern?',
#         'answer_template': 'The lesion is {value} symmetric.'
#     },
#     'Pigmented Reticulum': {
#         'question': 'Can you identify any pigmented reticulum?',
#         'answer_template': '{value} pigmented reticulum is observed.'
#     },
#     'Negative Reticulum': {
#         'question': 'Can you identify any negative reticulum?',
#         'answer_template': '{value} negative reticulum is observed.'
#     },
#     'Small Dots/Globules': {
#         'question': 'Can you identify any small dots/globules?',
#         'answer_template': '{value} small dots/globules are present.'
#     },
#     'Large Globules/Cobblestone': {
#         'question': 'Can you identify any large globules/cobblestone?',
#         'answer_template': '{value} large globules/cobblestone are observed.'
#     },
#     'Peripheral Globules': {
#         'question': 'Can you identify any peripheral globules?',
#         'answer_template': '{value} peripheral globules are observed.'
#     },
#     'Peripheral Projections': {
#         'question': 'Can you identify any peripheral projections?',
#         'answer_template': '{value} peripheral projections are present.'
#     },
#     'Homogeneous Area without Vessels': {
#         'question': 'Can you identify any homogeneous area without vessels?',
#         'answer_template': '{value} homogeneous area without vessels is observed.'
#     },
#     'Homogeneous Area with Vessels': {
#         'question': 'Can you identify any homogeneous area with vessels?',
#         'answer_template': '{value} homogeneous area with vessels is observed.'
#     },
#     'Undefined Area': {
#         'question': 'Can you identify any undefined area?',
#         'answer_template': '{value} undefined area is observed.'
#     },
#     'Gray Regression': {
#         'question': 'Can you identify any gray regression?',
#         'answer_template': '{value} gray regression is observed.'
#     },
#     'White Regression': {
#         'question': 'Can you identify any white regression?',
#         'answer_template': '{value} white regression is observed.'
#     },
#     'Whitish Blue Veil': {
#         'question': 'Can you identify any whitish blue veil?',
#         'answer_template': '{value} whitish blue veil is observed.'
#     },
#     'Chrysalis': {
#         'question': 'Can you identify any chrysalis?',
#         'answer_template': '{value} chrysalis is observed.'
#     },
#     'Binary label': {
#         'question': 'Is the lesion malignant?',
#         'answer_template': 'The lesion appears to be {value}.'
#     }
# }

qa_pairs = {
    'General Silhouette': {
        'question': [
            'Can you describe the general silhouette?',
            'How would you characterize the overall silhouette?',
            'What is the overall shape of the lesion?',
            'Describe the silhouette of the lesion.',
            'What does the silhouette of the lesion look like?'
        ],
        'answer_template': 'The general silhouette of the lesion is {value}.'
    },
    'Histology Diagnostic': {
        'question': [
            'What is the histology diagnostic?',
            'Could you provide the histological diagnosis?',
            'What does the histology indicate?',
            'Describe the histological diagnosis.',
            'Based on histology, what is the diagnosis?'
        ],
        'answer_template': 'The histology diagnostic indicates {value}.'
    },
    'Dysplasia': {
        'question': [
            'Can you see any dysplasia?',
            'Is there any evidence of dysplasia?',
            'Do you observe signs of dysplasia?',
            'Describe the presence of dysplasia.',
            'Is dysplasia present in the lesion?'
        ],
        'answer_template': 'The presence of dysplasia is {value}.'
    },
    'Diagnostic Impression': {
        'question': [
            'What is the diagnostic impression?',
            'Could you provide your impression of the diagnosis?',
            'What impression do you have regarding the diagnosis?',
            'Describe your overall diagnosis impression.',
            'What are your thoughts on the diagnosis?'
        ],
        'answer_template': 'The diagnostic impression is {value}.'
    },
    'Diagnostic Difficulty': {
        'question': [
            'What is the diagnostic difficulty?',
            'How challenging is the diagnosis?',
            'Can you assess the difficulty of the diagnosis?',
            'Describe the level of difficulty in the diagnosis.',
            'How difficult is it to make the diagnosis?'
        ],
        'answer_template': 'The diagnostic difficulty level is {value}.'
    },
    'Excision': {
        'question': [
            'Is there any excision?',
            'Has the lesion been excised?',
            'Was excision performed on the lesion?',
            'Describe any excision performed.',
            'Was the lesion removed by excision?'
        ],
        'answer_template': 'The lesion has been {value}.'
    },
    'Global Pattern': {
        'question': [
            'How is the global pattern?',
            'Can you describe the overall pattern?',
            'Describe the pattern observed globally.',
            'What is the pattern like throughout the lesion?',
            'What patterns are present in the entire lesion?'
        ],
        'answer_template': 'The global pattern of the lesion is {value}.'
    },
    'Peripheral Globular Pattern': {
        'question': [
            'How is the peripheral globular pattern?',
            'Describe the presence of peripheral globular pattern.',
            'Is there any peripheral globular pattern observed?',
            'What is the pattern like at the periphery?',
            'Are there globular patterns around the edges?'
        ],
        'answer_template': 'The peripheral globular pattern is {value}.'
    },
    'Symmetry': {
        'question': [
            'Is there any symmetry pattern?',
            'How would you describe the symmetry?',
            'Describe the symmetry of the lesion.',
            'Is the lesion symmetric or asymmetric?',
            'Can you assess the symmetry of the lesion?'
        ],
        'answer_template': 'The lesion is {value}.'
    },
    'Pigmented Reticulum': {
        'question': [
            'Can you identify any pigmented reticulum?',
            'Do you observe pigmented reticulum?',
            'Describe the presence of pigmented reticulum.',
            'Are there any signs of pigmented reticulum?',
            'What about pigmented reticulum in the lesion?'
        ],
        'answer_template': '{value} pigmented reticulum is observed.'
    },
    'Negative Reticulum': {
        'question': [
            'Can you identify any negative reticulum?',
            'Do you observe negative reticulum?',
            'Describe the presence of negative reticulum.',
            'Are there any signs of negative reticulum?',
            'What about negative reticulum in the lesion?'
        ],
        'answer_template': '{value} negative reticulum is observed.'
    },
    'Small Dots/Globules': {
        'question': [
            'Can you identify any small dots/globules?',
            'Do you see any small dots/globules?',
            'Describe the presence of small dots/globules.',
            'Are there any signs of small dots/globules?',
            'What about small dots/globules in the lesion?'
        ],
        'answer_template': '{value} small dots/globules are present.'
    },
    'Large Globules/Cobblestone': {
        'question': [
            'Can you identify any large globules/cobblestone?',
            'Do you observe large globules/cobblestone?',
            'Describe the presence of large globules/cobblestone.',
            'Are there any signs of large globules/cobblestone?',
            'What about large globules/cobblestone in the lesion?'
        ],
        'answer_template': '{value} large globules/cobblestone are observed.'
    },
    'Peripheral Globules': {
        'question': [
            'Can you identify any peripheral globules?',
            'Do you observe peripheral globules?',
            'Describe the presence of peripheral globules.',
            'Are there any signs of peripheral globules?',
            'What about peripheral globules in the lesion?'
        ],
        'answer_template': '{value} peripheral globules are observed.'
    },
    'Peripheral Projections': {
        'question': [
            'Can you identify any peripheral projections?',
            'Do you observe peripheral projections?',
            'Describe the presence of peripheral projections.',
            'Are there any signs of peripheral projections?',
            'What about peripheral projections in the lesion?'
        ],
        'answer_template': '{value} peripheral projections are present.'
    },
    'Homogeneous Area without Vessels': {
        'question': [
            'Can you identify any homogeneous area without vessels?',
            'Do you observe homogeneous area without vessels?',
            'Describe the presence of homogeneous area without vessels.',
            'Are there any signs of homogeneous area without vessels?',
            'What about homogeneous area without vessels in the lesion?'
        ],
        'answer_template': '{value} homogeneous area without vessels is observed.'
    },
    'Homogeneous Area with Vessels': {
        'question': [
            'Can you identify any homogeneous area with vessels?',
            'Do you observe homogeneous area with vessels?',
            'Describe the presence of homogeneous area with vessels.',
            'Are there any signs of homogeneous area with vessels?',
            'What about homogeneous area with vessels in the lesion?'
        ],
        'answer_template': '{value} homogeneous area with vessels is observed.'
    },
    'Undefined Area': {
        'question': [
            'Can you identify any undefined area?',
            'Do you observe undefined area?',
            'Describe the presence of undefined area.',
            'Are there any signs of undefined area?',
            'What about undefined area in the lesion?'
        ],
        'answer_template': '{value} undefined area is observed.'
    },
    'Gray Regression': {
        'question': [
            'Can you identify any gray regression?',
            'Do you observe gray regression?',
            'Describe the presence of gray regression.',
            'Are there any signs of gray regression?',
            'What about gray regression in the lesion?'
        ],
        'answer_template': '{value} gray regression is observed.'
    },
    'White Regression': {
        'question': [
            'Can you identify any white regression?',
            'Do you observe white regression?',
            'Describe the presence of white regression.',
            'Are there any signs of white regression?',
            'What about white regression in the lesion?'
        ],
        'answer_template': '{value} white regression is observed.'
    },
    'Whitish Blue Veil': {
        'question': [
            'Can you identify any whitish blue veil?',
            'Do you observe whitish blue veil?',
            'Describe the presence of whitish blue veil.',
            'Are there any signs of whitish blue veil?',
            'What about whitish blue veil in the lesion?'
        ],
        'answer_template': '{value} whitish blue veil is observed.'
    },
    'Chrysalis': {
        'question': [
            'Can you identify any chrysalis?',
            'Do you observe chrysalis?',
            'Describe the presence of chrysalis.',
            'Are there any signs of chrysalis?',
            'What about chrysalis in the lesion?'
        ],
        'answer_template': '{value} chrysalis is observed.'
    },
    'Binary label': {
        'question': [
            'Is the lesion malignant?',
            'Do you classify the lesion as malignant?',
            'What is the classification of the lesion?',
            'Is the lesion benign or malignant?',
            'Can you determine if the lesion is malignant?'
        ],
        'answer_template': 'The lesion appears to be {value}.'
    }
}

# Open CSV file and loop over rows
data = []
# Choose the necessary path in each case: melanoma_file_path or fusion_file_path
with open(fusion_file_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)

    # Iterate through each image
    for row in reader:

        image_id = row['File name (.jpg)']
        qa_pairs_for_image = []
        
        # Iterate through each variable for every image
        for variable, qa_pair in qa_pairs.items():

            # Iterate through every possible question per category
            for question in qa_pair['question']:

                # Save the template answer
                answer_template = qa_pair['answer_template']

                # The fusion data does not contain every variable in the melanoma one
                try:
                    actual_value = row[variable].lower()
                except KeyError:
                    continue

                # Insert the value into the template and capitalize the first word
                formatted_answer = answer_template.format(value=actual_value)
                formatted_answer = formatted_answer.split(' ')[0].capitalize() + ' ' + ' '.join(formatted_answer.split(' ')[1:])
                qa_pairs_for_image.append({'question': question, 'answer': formatted_answer})
        
        data.append({'image_id': image_id, 'qa_pairs': qa_pairs_for_image})

# Write data to JSON file
with open(fusion_output_path, 'w') as jsonfile:
    json.dump(data, jsonfile, indent=4)

