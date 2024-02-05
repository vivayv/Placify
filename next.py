import csv

def prepare_training_set(username, global_route):
    global topics, questions, answers, explanations, difficulty_levels, paraphrases, hints
    # Load the quiz dataset from CSV file
    topics = global_route # Modify the list of topics as per your requirement
    print(topics)
    questions = {}
    answers = {}
    explanations = {}
    difficulty_levels = {}
    paraphrases = {}
    hints = {}

    for topic in topics:
        questions[topic] = []
        answers[topic] = []
        explanations[topic] = []
        difficulty_levels[topic] = []
        paraphrases[topic] = []
        hints[topic] = []

        with open(f'{topic.replace(" ", "_")}.csv', 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                questions[topic].append(row['Question'])
                answers[topic].append(row['Answer'])
                explanations[topic].append(row['Explanation'])
                difficulty_levels[topic].append(row['Difficulty Level'])
                hints[topic].append(row['Hint'])
                print("Happened successfully")
                #sprint(hints)

                # Check if the paraphrase columns exist
                if 'Paraphrase 1' in row and 'Paraphrase 2' in row and 'Paraphrase 3' in row:
                    paraphrases[topic].append([row['Paraphrase 1'], row['Paraphrase 2'], row['Paraphrase 3']])
                    print("Paraphrase Happened successfully")
                elif 'Paraphrase 1' in row and 'Paraphrase 2' in row:
                    paraphrases[topic].append([row['Paraphrase 1'], row['Paraphrase 2']])
                    print("Paraphrase Happened successfully")
                elif 'Paraphrase 1' in row:
                    paraphrases[topic].append([row['Paraphrase 1']])
                    print("Paraphrase Happened successfully")
                else:
                    paraphrases[topic].append([])

        print('Everything happened in :')
        print(topic)

#global_route = ['Operating System','Object Oriented Programming','English Literature','Database Management System','C','Aptitude','Data Structures']
global_route = ['Computer Networks']
prepare_training_set('Sid',global_route)

'''
# Sample data: dictionary with historical data for each topic
data = {
    'Operating System': [5, 6, 7, 8, 7, 6, 5, 6, 7, 8],
    'Object Oriented Programming': [3, 4, 5, 6, 5, 4, 3, 4, 5, 6],
    'English Literature': [2, 3, 2, 4, 3, 2, 3, 4, 3, 2],
    'Database Management System': [7, 8, 6, 7, 8, 7, 8, 6, 7, 8],
    'Computer Networks': [4, 3, 5, 4, 3, 4, 5, 4, 3, 5],
    'C': [6, 8, 9, 7, 6, 8, 9, 7, 6, 8],
    'Aptitude': [4, 5, 6, 7, 8, 6, 5, 4, 3, 2],
    'Data Structures': [8, 7, 6, 5, 4, 5, 6, 7, 8, 7]
}

# Create a dictionary to store predicted values
predicted_data = {}

# Define a function to predict the next score using a linear regression model
def predict_next_score(scores):
    n = len(scores)
    x_mean = sum(range(1, n + 1)) / n
    y_mean = sum(scores) / n
    numer = sum((i - x_mean) * (score - y_mean) for i, score in enumerate(scores, 1))
    denom = sum((i - x_mean) ** 2 for i in range(1, n + 1))
    b = numer / denom
    a = y_mean - b * x_mean
    next_score = a + b * (n + 1)
    return int(next_score)

# Predict the next score for each topic and store in the dictionary
for topic, scores in data.items():
    next_score = predict_next_score(scores)
    predicted_data[topic] = next_score

# Print the dictionary with predicted values
print(predicted_data)

import Levenshtein

message = "by"

# List of reference strings
reference_strings = ['bye', 'goodbye']

# Check if the message is similar to any reference string based on Levenshtein distance
if any(Levenshtein.distance(message.lower(), ref) <= 10 for ref in ['bye', 'goodbye']) or message.lower() in ['bye', 'goodbye']:
    print(Levenshtein.distance(message.lower(), 'bye'))
    print("Message is similar to 'bye' or 'goodbye")
'''