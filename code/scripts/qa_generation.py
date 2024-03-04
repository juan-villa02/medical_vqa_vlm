import csv
import json

# Load CSV file
melanoma_file_path = '/Users/JUAN/Desktop/medical_vqa_vlm/data/processed/df_melanoma_clean.csv'
fusion_file_path = '/Users/JUAN/Desktop/medical_vqa_vlm/data/processed/df_fusion_clean.csv'

# JSON outputs
melanoma_output_path = '/Users/JUAN/Desktop/medical_vqa_vlm/data/processed//df_melanoma_qa.json'
fusion_output_path = '/Users/JUAN/Desktop/medical_vqa_vlm/data/processed//df_fusion_qa.json'

# Define initial question/answer format for each variable
qa_pairs = {
    'General Silhouette': {
        'question': 'Can you describe the general silhouette?',
        'answer_template': 'The general silhouette of the lesion is {value}.'
    },
    'Histology Diagnostic': {
        'question': 'What is the histology diagnostic?',
        'answer_template': 'The histology diagnostic indicates {value}.'
    },
    'Dysplasia': {
        'question': 'Can you see any dysplasia?',
        'answer_template': 'The presence of dysplasia is {value}.'
    },
    'Diagnostic Impression': {
        'question': 'What is the diagnostic impression?',
        'answer_template': 'The diagnostic impression is {value}.'
    },
    'Diagnostic Difficulty': {
        'question': 'What is the diagnostic difficulty?',
        'answer_template': 'The diagnostic difficulty level is {value}.'
    },
    'Excision': {
        'question': 'Is there any excision?',
        'answer_template': 'The lesion has been {value}.'
    },
    'Global Pattern': {
        'question': 'How is the global pattern?',
        'answer_template': 'The global pattern of the lesion is {value}.'
    },
    'Peripheral Globular Pattern': {
        'question': 'How is the peripheral globular pattern?',
        'answer_template': 'The peripheral globular pattern is {value}.'
    },
    'Symmetry': {
        'question': 'Is there any symmetry pattern?',
        'answer_template': 'The lesion is {value} symmetric.'
    },
    'Pigmented Reticulum': {
        'question': 'Can you identify any pigmented reticulum?',
        'answer_template': '{value} pigmented reticulum is observed.'
    },
    'Negative Reticulum': {
        'question': 'Can you identify any negative reticulum?',
        'answer_template': '{value} negative reticulum is observed.'
    },
    'Small Dots/Globules': {
        'question': 'Can you identify any small dots/globules?',
        'answer_template': '{value} small dots/globules are present.'
    },
    'Large Globules/Cobblestone': {
        'question': 'Can you identify any large globules/cobblestone?',
        'answer_template': '{value} large globules/cobblestone are observed.'
    },
    'Peripheral Globules': {
        'question': 'Can you identify any peripheral globules?',
        'answer_template': '{value} peripheral globules are observed.'
    },
    'Peripheral Projections': {
        'question': 'Can you identify any peripheral projections?',
        'answer_template': '{value} peripheral projections are present.'
    },
    'Homogeneous Area without Vessels': {
        'question': 'Can you identify any homogeneous area without vessels?',
        'answer_template': '{value} homogeneous area without vessels is observed.'
    },
    'Homogeneous Area with Vessels': {
        'question': 'Can you identify any homogeneous area with vessels?',
        'answer_template': '{value} homogeneous area with vessels is observed.'
    },
    'Undefined Area': {
        'question': 'Can you identify any undefined area?',
        'answer_template': '{value} undefined area is observed.'
    },
    'Gray Regression': {
        'question': 'Can you identify any gray regression?',
        'answer_template': '{value} gray regression is observed.'
    },
    'White Regression': {
        'question': 'Can you identify any white regression?',
        'answer_template': '{value} white regression is observed.'
    },
    'Whitish Blue Veil': {
        'question': 'Can you identify any whitish blue veil?',
        'answer_template': '{value} whitish blue veil is observed.'
    },
    'Chrysalis': {
        'question': 'Can you identify any chrysalis?',
        'answer_template': '{value} chrysalis is observed.'
    },
    'Binary label': {
        'question': 'Is the lesion malignant?',
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

            # Save the formatted question
            question = qa_pair['question']
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

