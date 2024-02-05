from flask import Flask, render_template, request, session, redirect, url_for
from flask import jsonify
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.metrics import edit_distance
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import pandas as pd
import tflearn
import tensorflow as tf
import random
import json
import csv
import secrets
from difflib import SequenceMatcher
import pyodbc
from PyPDF2 import PdfReader
import re
import datetime
#from sklearn.linear_model import LinearRegression
import plotly.express as px
import Levenshtein
import difflib

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Set the server, database, and driver details
server = 'VIVA-LAPTOP'
database = 'capstone'
driver = '{SQL Server}'

# Create the connection string with Windows Authentication
conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes;'

# Connect to the SQL Server database
try:
    conn = pyodbc.connect(conn_str)
    print('Connection successful')
except pyodbc.Error as e:
    print(f'Error connecting to database: {e}')

# Execute a query to fetch the student grades
query = "SELECT * FROM student_grades"
grades_df = pd.read_sql_query(query, conn)
grades_df = grades_df.set_index('Name')

# Print the dataframe
print(grades_df)

cursor = conn.cursor()

nltk.download('stopwords')
nltk.download('punkt')

# Define the states
location_to_state = {
  'Operating System': 0,
  'Object Oriented Programming': 1,
  'English Literature' : 2,
  'Database Management System' : 3,
  'Computer Networks' : 4,
  'C' : 5,
  'Aptitude' : 6,
  'Data Structures' : 7
}

location_list = ['Operating System','Object Oriented Programming','English Literature','Database Management System','Computer Networks','C','Aptitude','Data Structures']

actions = [0,1,2,3,4,5,6,7]

rewards = np.array([[-1, 2, -1, 2, 4, 1, -1, 1],
              [1, -1, -1, 2, 1, 3, 1, 3],
              [1, 1, -1, 1, 1, 1, 3, 1],
              [2, 3, -1, -1, 2, 1, -1, 2],
              [4, 2, -1, 1, -1, 1, -1, 2],
              [3, -1, -1, 1, 2, -1, 1, 4],
              [1, 1, -1, 1, 1, 1, -1, 2],
              [2, 3, -1, 1, 1, 4, 2, -1]])

# Maps indices to Locations
state_to_location = dict((state, location) for location, state in location_to_state.items())

# Initialize parameters
gamma = 0.2 # Discount factor
alpha = 0.9 # Learning rate

class QAgent ():
  # Initialize alpha, gamma, states, actions, rewards, and Q-values
  def __init__(self, alpha, gamma, location_to_state, actions, rewards, state_to_location):
    self.gamma = gamma
    self.alpha = alpha
    self.location_to_state = location_to_state
    self.actions = actions
    self.rewards = rewards
    self.state_to_location = state_to_location
    self.num_states = len(location_to_state)

    M = len(location_to_state)
    self.Q = np.zeros((M,M),dtype = None, order = 'C')

  # Training the robot in the environment
  def training(self, start_location, end_location, iterations):
    rewards_new = np.copy(self.rewards)
    ending_state= self.location_to_state[end_location] 
    rewards_new[ending_state, ending_state] = 999
    #picking a random current state
    for i in range(iterations):
      current_state = self.location_to_state[start_location]
      playable_actions = [state for state in range(self.num_states) if rewards_new[current_state, state] >= 0]
      if not playable_actions:
        continue
      # iterate through the rewards matrix to get the states
      # direc   tly reachable from the randomly chosen current state # assign those state in a list named playable_actions.
      '''
      for j in range(8):
        if rewards_new[current_state, j] > 0:
          playable_actions.append(j)
      '''
      #choosing a random next state
      next_state = np.random.choice(playable_actions)

      #finding temporal difference
      TD = rewards_new[current_state,next_state] + self.gamma * self.Q[next_state,np.argmax(self.Q[next_state,])] - self.Q[current_state,next_state]

      self.Q[current_state,next_state] += self.alpha * TD

    route = [start_location]
    next_location = start_location

    #print('this happeneed 1')
    #Get The Route
    self.get_optimal_route(start_location, end_location, next_location, route, self.Q)
    #print('this happened 3')

  # Get the optimal route
  def get_optimal_route(self, start_location, end_location, next_location, route, Q):
    while(next_location != end_location):
      starting_state = self.location_to_state[start_location] 
      next_state = np.argmax(Q[starting_state,])
      next_location= self.state_to_location[next_state]
      route.append(next_location)
      start_location = next_location
    #print('this happeneed 2')
    print(route)
    global_route.extend([x for x in route if x not in global_route])
    print(global_route)

global_route = []
#qagent = QAgent(alpha,gamma,location_to_state,actions,rewards,state_to_location)
#print(global_route)


stemmer = LancasterStemmer()
words = []
labels = []
docs_x = []
docs_y = []

with open("intents.json") as file:
    data = json.load(file)

for intent in data['intents']:
    for pattern in intent['patterns']:
        pattern = pattern.lower()
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]
for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Create a new graph
graph = tf.Graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch=100, batch_size=8, show_metric=True)
model.save("model.tflearn")

# Load the saved model
model.load("model.tflearn")

topics = global_route
questions = {}
answers = {}
explanations = {}
difficulty_levels = {}
paraphrases = {}
hints = {}
user_progress = {}

def prepare_training_set(username, global_route):
    global topics, questions, answers, explanations, difficulty_levels, paraphrases, user_progress, hints
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
                #print("Happened successfully")
                #sprint(hints)

                # Check if the paraphrase columns exist
                if 'Paraphrase 1' in row and 'Paraphrase 2' in row and 'Paraphrase 3' in row:
                    paraphrases[topic].append([row['Paraphrase 1'], row['Paraphrase 2'], row['Paraphrase 3']])
                elif 'Paraphrase 1' in row and 'Paraphrase 2' in row:
                    paraphrases[topic].append([row['Paraphrase 1'], row['Paraphrase 2']])
                elif 'Paraphrase 1' in row:
                    paraphrases[topic].append([row['Paraphrase 1']])
                else:
                    paraphrases[topic].append([])

    for topic in topics:
        cursor.execute("SELECT current_question_index, current_difficulty FROM user_progress WHERE username = ? AND topic = ?", (username, topic))
        row = cursor.fetchone()
        if row is not None:
            user_progress[topic] = {
                'asked_questions': [],
                'current_question_index': row[0],
                'current_difficulty': row[1]
            }
        else:
            user_progress[topic] = {
                'asked_questions': [],
                'current_question_index': -1,
                'current_difficulty': 'Easy'
            }

# Function to calculate similarity between two words using Levenshtein distance
def is_similar(word1, word2):
    return Levenshtein.distance(word1.lower(), word2.lower()) <= 0.5

def extract_topics(input_text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(input_text)

    # removing stop words and punctuation
    filtered_tokens = [w for w in word_tokens if w.lower() not in stop_words and w.isalpha()]

    # generating bigrams and trigrams
    bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))

    found_subjects = set()  # Use a set to avoid duplicates

    # searching for tokens
    for token in filtered_tokens:
        for subject, keywords in subject_keywords.items():
            if token in keywords:
                found_subjects.add(subject)

    # searching for bigrams and trigrams
    for ngram in bigrams_trigrams:
        for subject, keywords in subject_keywords.items():
            if ngram in keywords:
                found_subjects.add(subject)

    # searching for similar words to known topics
    for token in filtered_tokens:
        for subject, keywords in subject_keywords.items():
            for keyword in keywords:
                if is_similar(token, keyword):
                    found_subjects.add(subject)

    # searching for variations of known topics in the resume text
    for subject, keywords in subject_keywords.items():
        for keyword in keywords:
            variations_regex = r'\b(?:{})\b'.format('|'.join(re.escape(w) for w in keyword.lower().split()))
            if re.search(variations_regex, input_text.lower()):
                found_subjects.add(subject)

    found_subjects = list(found_subjects)  # Convert set to list
    return found_subjects

def bag_of_words(sentence, words):
    bag = [0 for _ in range(len(words))]
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return bag

def get_response_from_label(label):
    for intent in data['intents']:
        if intent['tag'] == label:
            return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that."

def is_answer_similar(user_answer, correct_answer, paraphrases):
    similarity_ratio = SequenceMatcher(None, user_answer.lower(), correct_answer.lower()).ratio()

    if paraphrases:
        for paraphrase in paraphrases:
            if SequenceMatcher(None, user_answer.lower(), paraphrase.lower()).ratio() > similarity_ratio:
                similarity_ratio = SequenceMatcher(None, user_answer.lower(), paraphrase.lower()).ratio()

    return similarity_ratio >= 0.42  # Adjust the similarity threshold as per your requirement

def get_similarity_score(user_answer, correct_answer, paraphrases):
    similarity_score = SequenceMatcher(None, user_answer.lower(), correct_answer.lower()).ratio()

    if paraphrases:
        for paraphrase in paraphrases:
            if SequenceMatcher(None, user_answer.lower(), paraphrase.lower()).ratio() > similarity_score:
                similarity_score = SequenceMatcher(None, user_answer.lower(), paraphrase.lower()).ratio()

    return similarity_score # Adjust the similarity threshold as per your requirement

# Function to authenticate the user against the user_details table
def authenticate_user(username, password):
    try:
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            # Execute a SQL query to retrieve the user with the provided username and password
            cursor.execute("SELECT * FROM user_details WHERE username = ? AND password = ?", (username, password))
            user = cursor.fetchone()
            if user:
                return True
    except pyodbc.Error as e:
        print(f"Database error: {str(e)}")
    return False

def extract_text_from_resume(resume_file):
    # Assuming the resume file is in PDF format
    pdf_reader = PdfReader(resume_file)
    resume_text = ""
    for page in pdf_reader.pages:
        resume_text += page.extract_text()
    return resume_text

def convert_to_letter_grade(numeric_grade, grade_ranges):
    # Convert numeric grades to letter grades based on the given grade ranges
    for grade_range, letter_grade in grade_ranges.items():
        if grade_range[0] <= numeric_grade <= grade_range[1]:
            return letter_grade
    return 'Invalid' 



import re

subject_keywords = dict()
subject_keywords['Database Management System'] = ['dbms', 'database', 'databases', 'database management', 'database management systems', 'database systems', 'SQL', 'Relational databases', 'NoSQL', 'Query language', 'Data modeling', 'Database administration', 'Indexing', 'Normalization', 'ACID properties', 'Data warehousing', 'Data retrieval', 'Data integrity', 'Primary key', 'Foreign key', 'Data security']
subject_keywords['Object Oriented Programming'] = ['oops', 'object oriented programming', 'object oriented programming systems', 'object oriented design', 'Classes', 'Inheritance', 'Polymorphism', 'Encapsulation', 'Abstraction', 'Constructors', 'Overloading', 'Interfaces', 'UML (Unified Modeling Language)', 'Class hierarchy', 'Method overriding', 'Composition', 'Design patterns', 'Object oriented analysis and design']
subject_keywords['Data Structures'] = ['dsa', 'data structures', 'data structures and algorithms', 'algorithms', 'Linked lists', 'Stacks', 'Queues', 'Trees', 'Graphs', 'Hash tables', 'Arrays', 'Linked lists', 'Heaps', 'Trie', 'AVL trees', 'Binary search', 'Priority queues', 'Hashing', 'Red-Black trees']
subject_keywords['Computer Networks'] = ['cn', 'computer networks', 'networks', 'packets', 'packet switching', 'router', 'routers', 'switch', 'switches', 'protocols', 'IP', 'TCP', 'LAN (Local Area Network)', 'WAN (Wide Area Network)', 'Subnetting', 'DNS (Domain Name System)', 'Firewall', 'Network topology', 'OSI model', 'Network protocols', 'Network security', 'Network administration', 'Network architecture', 'Ethernet', 'Subnet masks', 'DHCP (Dynamic Host Configuration Protocol)', 'Routing tables']
subject_keywords['Operating System'] = ['os', 'operating system', 'operating systems', 'linux', 'terminal', 'kernel', 'Process management', 'Memory management', 'File systems', 'Virtual memory', 'Multithreading', 'Scheduling algorithms', 'Deadlock handling', 'System calls', 'Shell scripting', 'Device drivers', 'Disk management', 'User permissions', 'Interrupt handling', 'Kernel modules', 'Boot process']

def extract_subjects(resume_text):
    # Initialize a set to store subjects/topics without duplicates
    subjects_set = set()

    # Convert the resume text to lowercase for case-insensitive matching
    resume_text = resume_text.lower()

    # Use regular expressions to find matches for subject keywords
    for keyword in subject_keywords:
        # Create a regular expression pattern for the keyword
        for subkeyword in subject_keywords[keyword]:
            pattern = re.compile(r'\b' + re.escape(subkeyword) + r'\b')
            
            # Find all matches in the text
            matches = pattern.findall(resume_text)

            # If matches are found, add the keyword to the subjects set
            if matches:
                subjects_set.add(keyword)

    # Convert the set to a list to maintain the order
    subjects_list = list(subjects_set)
    
    print(subjects_list)
    return subjects_list



def calculate_weighted_scores(score_tracking_dict, number_qns_dict, found_strong_topics, location_list):
    weighted_scores = {}

    for subject in location_list:
        if subject in score_tracking_dict and subject in number_qns_dict:
            accuracy = score_tracking_dict[subject]
            num_questions = number_qns_dict[subject]

            # Define a weight to balance accuracy and the number of questions answered
            weight = 0.5  # You can adjust this weight based on the importance you give to accuracy vs. the number of questions answered

            # If the subject is in found_strong_topics, assign an additional weight
            if subject in found_strong_topics:
                weight += 0.2  # You can adjust this additional weight

            # Calculate the weighted score
            weighted_score = (accuracy * (1 - weight)) + (num_questions * weight)
            weighted_scores[subject] = weighted_score

    return weighted_scores

def suggest_roles(weighted_scores, threshold=0.7):
    # Customize this list with tech and consulting roles
    roles = {
        'Software Developer': ['Operating System', 'Object Oriented Programming', 'Data Structures'],
        'Database Administrator': ['Database Management System'],
        'Network Engineer': ['Computer Networks'],
        'Consultant': ['Aptitude'],
        'Web Developer': ['Object Oriented Programming', 'Data Structures'],
        'Front-End Developer': ['Object Oriented Programming', 'Data Structures', 'English Literature'],
        'Back-End Developer': ['Object Oriented Programming', 'Database Management System', 'Operating System'],
        'Mobile App Developer': ['Object Oriented Programming', 'Database Management System', 'Data Structures'],
        'Game Developer': ['Object Oriented Programming', 'Data Structures', 'C'],
        'Data Analyst': ['Database Management System', 'Data Structures'],
        'Cloud Engineer': ['Database Management System', 'Operating System', 'Computer Networks'],
        'Security Analyst': ['Operating System', 'Computer Networks', 'Data Structures'],
        'IT Consultant': ['Aptitude', 'Object Oriented Programming'],
        'Software Quality Assurance Engineer': ['Object Oriented Programming', 'Database Management System', 'Aptitude'],
        'Database Developer': ['Database Management System', 'Object Oriented Programming', 'Data Structures'],
        'Network Administrator': ['Computer Networks', 'Operating System', 'Aptitude'],
        'Machine Learning Engineer': ['Object Oriented Programming', 'Data Structures', 'Database Management System', 'Aptitude'],
        'UI/UX Designer': ['Object Oriented Programming', 'English Literature', 'Aptitude']
    }

    suggested_roles = []
    sorted_scores = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
    for subject, score in sorted_scores:
        for role, required_subjects in roles.items():
            if subject in required_subjects and score >= threshold and role not in suggested_roles:
                suggested_roles.append(role)
                break  # Move to the next subject

        if len(suggested_roles) >= 4:
            break  # Limit to the top 4 suggested roles

    return suggested_roles


def classify_subjects(weighted_scores):
    proficiency_levels = {}
    for subject, score in weighted_scores.items():
        print(score)
        if isinstance(score, (int, float)):
            converted_score = int(score)
            if converted_score >= 6:
                proficiency_levels[subject] = 'Expert'
            elif 5 <= converted_score < 6:
                proficiency_levels[subject] = 'Advanced'
            elif 4 <= converted_score < 5:
                proficiency_levels[subject] = 'Well-Versed'
            elif 3 <= converted_score < 4:
                proficiency_levels[subject] = 'Good'
            else:
                proficiency_levels[subject] = 'Proficient'
        else:
            proficiency_levels[subject] = score  # Preserve the original value if it's not an integer or float
    return proficiency_levels


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

def create_plot(title, data_dict):
    data = {'Topic': list(data_dict.keys()), 'Value': list(data_dict.values())}
    df = pd.DataFrame(data)
    fig = px.bar(df, x='Topic', y='Value', title=title)
    return fig.to_html(full_html=False)

def calculate_suggestions(score_tracking_dict, number_qns_dict, subj_correct_dict, subj_total_dict):
    suggestions = []

    for topic in score_tracking_dict:
        # Check if the topic is present in all dictionaries
        if topic in number_qns_dict and topic in subj_correct_dict and topic in subj_total_dict:
            accuracy = score_tracking_dict[topic]
            num_training_questions = number_qns_dict[topic]
            num_correct_test_answers = subj_correct_dict[topic]
            num_total_test_questions = subj_total_dict[topic]

            # Your logic for generating suggestions goes here
            if accuracy>0.3 and accuracy <= 0.5:
                suggestions.append(f"Consider reviewing {topic}. Your accuracy in training was quite low. Try making your fundamentals in the subject more strong.")
            elif accuracy<=0.3:
                suggestions.append(f"{topic} seems challenging for you. Break down the subject into simpler concepts and practice more.")
            elif accuracy>0.5 and accuracy<0.75:
                suggestions.append(f"Your performance {topic} was not bad, but you could go a bit more in depth into the subject to become an expert at it.")

            if (num_correct_test_answers / num_total_test_questions)>0.3 and (num_correct_test_answers / num_total_test_questions) <= 0.5:
                suggestions.append(f"Focus on improving your performance in {topic}. Your accuracy in tests is below average. Practise the easy and medium level questions more to shore up your fundamentals.")
            elif (num_correct_test_answers / num_total_test_questions)<=0.3:
                suggestions.append(f"Train yourself more in the fundamentals in {topic}. You could use a lot more improvement in this subject.")
            elif (num_correct_test_answers / num_total_test_questions)>0.5 and (num_correct_test_answers / num_total_test_questions)<0.75:
                suggestions.append(f"While you haved passed the tests in {topic}, you can train more in hard questions to become an expert in the subject.")

            # You can add more conditions and suggestions based on your requirements

    return suggestions


certified_topics = []
subjects_list = []
strong_swot_list = []
weak_swot_list = []
found_strong_topics = []
global_weighted_scores = {}
location_dict = {}

for location in location_list:
    location_dict[location] = []

weak_reward_increase = 3
strong_reward_increase = 1


@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('chatbot'))
    else:
        return redirect(url_for('register'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        global subjects_list
        # Process the registration form data
        username = request.form['username']
        password = request.form['password']

        # Check if the username already exists
        query = "SELECT username FROM user_details WHERE username = ?"
        cursor.execute(query, (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            error_message = "Username already exists. Please choose a different username."
            return render_template('register.html', error_message=error_message)

        # Store user details in the database
        query = "INSERT INTO user_details (username, password) VALUES (?, ?)"
        cursor.execute(query, (username, password))
        conn.commit()
    
        # Extract text from the uploaded resume
        resume_file = request.files['resume']
        if resume_file:
            resume_text = extract_text_from_resume(resume_file)
        
        #print(resume_text)
        #found_topics = extract_topics(resume_text)
        #print(found_topics)

        subjects_list = extract_subjects(resume_text)
       
        print(resume_text)

        print(subjects_list)
        
        # Redirect the user to the login page after successful registration
        session['username'] = username
        print(session['username'])
        return redirect(url_for('swot_analysis'))

    # Redirect to the login page if the user is already registered
    if 'username' in session:
        return redirect(url_for('login'))
        
    # Render the SWOT analysis form template for GET requests
    return render_template('register.html')

@app.route('/swot_analysis', methods=['GET', 'POST'])
def swot_analysis():
    if request.method == 'POST':
        global strong_swot_list, weak_swot_list
        strengths = request.form['strengths']
        weaknesses = request.form['weaknesses']
        opportunities = request.form['opportunities']
        threats = request.form['threats']

        # Now you have the SWOT analysis data
        # You can perform any necessary processing with the SWOT analysis here
        strengths_text = ""
        weaknesses_text = ""
        strengths_text = strengths_text.join(strengths)
        weaknesses_text = weaknesses_text.join(weaknesses)
        strong_swot_list = extract_subjects(strengths_text)
        weak_swot_list = extract_subjects(weaknesses_text)
        print('happened')

        # Redirect to the grades form
        return redirect(url_for('grades'))

    # Render the SWOT analysis form template for GET requests
    return render_template('swotmatrix.html')

@app.route('/grades', methods=['GET', 'POST'])
def grades():
    global global_route, found_strong_topics  # Add this line to access the global variable
    if 'username' not in session:
        global_route = []  # Reset global_route for a new user session
        return redirect(url_for('index'))
    
    username = session['username']

    if request.method == 'POST':
        # Process the grade form data
        dbms_grade = request.form['dbms']
        dsa_grade = request.form['dsa']
        oops_grade = request.form['oops']
        cn_grade = request.form['cn']
        os_grade = request.form['os']
        c_grade = dsa_grade
        eng_grade = 'A'
        apt_grade = 'C'

        # Check the user's grade input option
        grade_option = request.form['grade_option']

        # Convert numeric grades if selected
        if grade_option == 'numbers':
            # Define grade ranges
            grade_ranges = {
                (90, 100): 'S',
                (75, 89.99): 'A',
                (60, 74.99): 'B',
                (50, 59.99): 'C',
                (40, 49.99): 'D',
                (28, 39.99): 'E',
                (0, 27.99): 'F',
            }

            # Convert the grades
            dbms_grade = convert_to_letter_grade(float(dbms_grade), grade_ranges)
            dsa_grade = convert_to_letter_grade(float(dsa_grade), grade_ranges)
            oops_grade = convert_to_letter_grade(float(oops_grade), grade_ranges)
            cn_grade = convert_to_letter_grade(float(cn_grade), grade_ranges)
            os_grade = convert_to_letter_grade(float(os_grade), grade_ranges)

        # Now you have the grades for each subject
        # You can perform any necessary processing with the grades here

        # Continue with the rest of your code to extract text from the resume,
        # calculate grades, and perform other actions

        # Create a dictionary to store modified column names and grades
        grades_dict = {
            'Database Management System': dbms_grade,
            'Data Structures': dsa_grade,
            'Object Oriented Programming': oops_grade,
            'Operating System': os_grade,
            'Computer Networks': cn_grade,
            'English Literature': eng_grade,
            'C': c_grade,
            'Aptitude': apt_grade
        }

        # Print the grades dictionary
        print(grades_dict)

        if len(grades_dict) > 0:
            # Classify grades as strong, neutral, or weak based on user-defined strengths and weaknesses
            strong_topics = []
            weak_topics = []
            for topic, grade in grades_dict.items():
                # Check if the subject is in the list of strengths
                if topic in strong_swot_list:
                    if topic in subjects_list:
                        if grade in ['S','A','B','C']:
                            strong_topics.append(topic)
                        elif grade in ['D','E','F']:
                            weak_topics.append(topic)
                    elif topic not in strong_swot_list:
                        if grade in ['S','A','B']:
                            strong_topics.append(topic)
                        elif grade in ['C','D','E','F']:
                            weak_topics.append(topic)
                # Check if the subject is in the list of weaknesses
                elif topic in weak_swot_list:
                    weak_topics.append(topic)
                # Check if the subject was discovered in the resume or not
                elif topic in subjects_list:
                    if grade in ['S','A','B']:
                        strong_topics.append(topic)
                    elif grade in ['C','D','E','F']:
                        weak_topics.append(topic)
                # If not found in strengths or weaknesses, classify based on grades
                elif grade in ['S', 'A']:
                    strong_topics.append(topic)
                elif grade in ['B', 'C', 'D', 'E', 'F']:
                    weak_topics.append(topic)

            # Update the reward matrix based on weak and strong topics
            for i in range(len(location_list)):
                for j in range(len(location_list)):
                    if i != j: # Check if i and j are different (not transitioning to the same state)
                        if state_to_location[i] in weak_topics:
                            rewards[i][j] += weak_reward_increase
                        elif state_to_location[i] in strong_topics:
                            rewards[i][j] += strong_reward_increase
            
            found_strong_topics = strong_topics
            print(strong_topics)
            print(weak_topics)
            print(rewards)

            qagent = QAgent(alpha,gamma,location_to_state,actions,rewards,state_to_location)

            if len(strong_topics) > 0 and len(weak_topics) > 0:
                # Choose a strong topic and a weak topic
                strong_topic = random.sample(strong_topics, 1)
                weak_topic = random.sample(weak_topics, 1)
                strong_topic = strong_topic[0]
                weak_topic = weak_topic[0]

                global_route = []

                # Call qagent.training with the chosen elements
                qagent.training(weak_topic, strong_topic, 1000)

                '''
                while len(global_route) != len(location_list):
                    temp_list = location_list[:]  # Create a copy of location_list
                    for item in global_route:
                        if item in temp_list:
                            temp_list.remove(item)
                    if len(temp_list) > 0:  # Check if temp_list is not empty
                        random_1 = random.choice(temp_list)
                        random_2 = random.choice(temp_list)
                        qagent.training(random_1, random_2, 1000)
                    else:
                        break
                '''
                # Continue to connect the topics until all topics are in the route
                missing_topics = [topic for topic in location_list if topic != 'English Literature']
                for topic in missing_topics:
                    if topic not in global_route:
                        global_route.append(topic)
                global_route.append('English Literature')

                print(global_route)
                prepare_training_set(username, global_route)

                # Get the optimal route
                optimal_route = global_route
                route_str = ', '.join(optimal_route)

                # Store the string in the personalized_routes table
                query = "INSERT INTO personalized_routes (username, route) VALUES (?, ?)"
                cursor.execute(query, (username, route_str))
                conn.commit()

        # Redirect the user to the login page after successful registration
        return redirect(url_for('login'))

    # Render the grade input form template for GET requests
    return render_template('getgrades.html')

@app.route('/login', methods=['GET','POST'])
def login():
    global global_route, certified_topics
    if 'username' in session:
        return redirect(url_for('chatbot'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if authenticate_user(username, password):
            session['username'] = username
            # Assuming you have the username stored in the 'username' variable
            cursor.execute("SELECT route FROM personalized_routes WHERE username = ?", (username))
            row = cursor.fetchone()

            if row is not None:
                topics_text = row[0]
            else:
                # Handle the case when no row is found for the given username
                # For example, set topics_text to an empty string or handle the error condition
                topics_text = ""

            topics = topics_text.split(', ')
            global_route = topics
            prepare_training_set(username, topics)

            # Execute a SQL query to fetch the 'certified_topics' column for the user 'Sid'
            cursor.execute("SELECT certified_topics FROM test_progress WHERE username = ?", (username))
            row = cursor.fetchone()
            # Check if the 'certified_topics' column is empty or not
            if row is not None and row[0]:  # row[0] contains the value of 'certified_topics'
                # 'certified_topics' column is not empty for the user
                certified_topics = row[0].split(',')
                print(certified_topics)
            else:
                certified_topics = []
                print(certified_topics)

            return redirect(url_for('chatbot'))
        else:
            return render_template('login.html', error_message='Invalid username or password')
    
    # Render the login form template for GET requests
    error_message=None
    return render_template('login.html',error_message=error_message)

num_correct = 0
asked_questions = 0
each_subject_correct = 0
each_subject_asked = 0
check = -1
score_tracking_dict = {}
number_qns_dict = {}
subj_correct_dict = {}
subj_total_dict = {}

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    global global_route, num_correct, asked_questions, each_subject_asked, each_subject_correct, check, global_weighted_scores, location_dict
    if 'username' not in session:
        global_route = []  # Reset global_route for a new user session
        return redirect(url_for('index'))
    
    username = session['username']  # Retrieve the username from the session

    # Assuming you have the username stored in the 'username' variable
    cursor.execute("SELECT route FROM personalized_routes WHERE username = ?", (username))
    row = cursor.fetchone()

    if row is not None:
        topics_text = row[0]
    else:
        # Handle the case when no row is found for the given username
        # For example, set topics_text to an empty string or handle the error condition
        topics_text = ""

    topics = topics_text.split(',')
    print(topics)
    print(len(topics))

    if request.method == 'POST':
        message = request.form['message']
        user_topic = session.get('topic', '').strip()
        user_progress_topic = user_progress[user_topic]
        session['progress'] = user_progress
        print(user_topic)

        for topic, progress in user_progress.items():
            #cursor.execute("INSERT OR REPLACE INTO user_progress (username, topic, current_question_index, current_difficulty) VALUES (?, ?, ?, ?)", (username, topic, progress['current_question_index'], progress['current_difficulty']))
            cursor.execute("""
                MERGE INTO user_progress AS target
                USING (VALUES (?, ?, ?, ?)) AS source (username, topic, current_question_index, current_difficulty)
                ON (target.username = source.username AND target.topic = source.topic)
                WHEN MATCHED THEN
                    UPDATE SET
                        current_question_index = source.current_question_index,
                        current_difficulty = source.current_difficulty
                WHEN NOT MATCHED THEN
                    INSERT (username, topic, current_question_index, current_difficulty)
                    VALUES (source.username, source.topic, source.current_question_index, source.current_difficulty)
                ;
            """, (username, topic, progress['current_question_index'], progress['current_difficulty']))
        conn.commit()

        if any(Levenshtein.distance(message.lower(), ref) <= 1 for ref in ['bye', 'goodbye']) or message.lower() in ['bye', 'goodbye']:
            session.pop('topic', None)
            session.pop('username', None)
            session.pop('progress', None)
            return render_template('chatbot.html', message=random.choice(data['intents'][5]['responses']), topic=user_topic)
        
        #if message.lower() in ["Let's talk","Let's have a conversation","Let's chat"]:
        if any(Levenshtein.distance(message.lower(), ref) <= 1 for ref in ["Let's talk","Let's have a conversation","Let's chat"]) or message.lower() in ["Let's talk","Let's have a conversation","Let's chat"]:
            return render_template('chatbot.html', message=random.choice(data['intents'][6]['responses']), topic=user_topic)

        if 'topic' not in session:
            return redirect(url_for('index'))

        if request.method == 'POST':
            message = request.form['message']
            user_topic = session.get('topic', '').strip()
            user_progress_topic = user_progress[user_topic]

            if any(Levenshtein.distance(message.lower(), ref) <= 1 for ref in ['bye', 'goodbye']) or message.lower() in ['bye', 'goodbye']:
                session.pop('topic', None)
                session.pop('username', None)
                session.pop('progress', None)
                return render_template('chatbot.html', message=random.choice(data['intents'][5]['responses']), topic=user_topic)
            
            #if message.lower() in ['can i get a hint', 'hint']:
            if any(Levenshtein.distance(message.lower(), ref) <= 1 for ref in ['can i get a hint', 'hint']) or message.lower() in ['can i get a hint', 'hint']:
                current_question_index = user_progress_topic['current_question_index']
                current_difficulty = user_progress_topic['current_difficulty']
                questions_of_current_difficulty = [
                    i for i, level in enumerate(difficulty_levels[user_topic]) if level == current_difficulty
                ]
                if current_question_index < len(questions_of_current_difficulty):
                    current_question = questions[user_topic][questions_of_current_difficulty[current_question_index]]
                    hint = hints.get(user_topic.strip(), [])[current_question_index]

                    if hint and hint != "There are no hints.":
                        return render_template('chatbot.html', message=f"Here's a hint for your current question:\nHint: {hint}\n\nQuestion: {current_question}", topic=user_topic)
                    else:
                        return render_template('chatbot.html', message=f"I'm sorry, but there's no hint available for the current question:\n\nQuestion: {current_question}", topic=user_topic)
                else:
                    return render_template('chatbot.html', message="I'm sorry, but there's no question available for a hint at the moment.", topic=user_topic)

            #if message.lower() in ['okay', 'thanks', 'next']:
            if any(Levenshtein.distance(message.lower(), ref) <= 1 for ref in ['okay', 'thanks', 'next']) or message.lower() in ['okay', 'thanks', 'next']:
                current_question_index = user_progress_topic['current_question_index']
                current_difficulty = user_progress_topic['current_difficulty']

                # Get the questions of the current difficulty level
                questions_of_current_difficulty = [
                    i for i, level in enumerate(difficulty_levels[user_topic]) if level == current_difficulty
                ]

                # Check if there are more questions of the current difficulty level
                if current_question_index < len(questions_of_current_difficulty) - 1:
                    print(current_question_index)
                    print(len(questions_of_current_difficulty))
                    print(asked_questions)
                    each_subject_correct = each_subject_correct + num_correct
                    each_subject_asked = each_subject_asked + asked_questions
                    if asked_questions<2:
                        user_progress_topic['current_question_index'] += 1
                        current_question_index = user_progress_topic['current_question_index']
                        current_question = questions[user_topic][questions_of_current_difficulty[current_question_index]]
                        return render_template('chatbot.html', message=f"Great! Here's your next question:\n{current_question}", topic=user_topic)
                    else:
                        if num_correct/asked_questions <= 0.7:
                            location_dict[user_topic].append(num_correct)
                            print(location_dict)
                            asked_questions = 0
                            num_correct = 0
                            user_progress_topic['current_question_index'] += 1
                            current_question_index = user_progress_topic['current_question_index']
                            current_question = questions[user_topic][questions_of_current_difficulty[current_question_index]]
                            return render_template('chatbot.html', message=f"Great! Here's your next question:\n{current_question}", topic=user_topic)
                        else:
                            location_dict[user_topic].append(num_correct)
                            print(location_dict)
                            asked_questions = 0
                            num_correct = 0
                            # Check if there are more difficulty levels remaining
                            if current_difficulty == 'Easy':
                                if 'Medium' in difficulty_levels[user_topic]:
                                    user_progress_topic['current_difficulty'] = 'Medium'
                                    current_difficulty = user_progress_topic['current_difficulty']
                                elif 'Hard' in difficulty_levels[user_topic]:
                                    user_progress_topic['current_difficulty'] = 'Hard'
                                    current_difficulty = user_progress_topic['current_difficulty']
                            elif current_difficulty == 'Medium':
                                if 'Hard' in difficulty_levels[user_topic]:
                                    user_progress_topic['current_difficulty'] = 'Hard'
                                    current_difficulty = user_progress_topic['current_difficulty']
                            else:
                                # Check if there are more topics remaining
                                print('Before check 1')
                                print(len(topics))
                                if len(topics) == 1:
                                    check = 1
                                    weighted_scores = calculate_weighted_scores(score_tracking_dict, number_qns_dict, found_strong_topics,location_list)
                                    suggested_roles = suggest_roles(weighted_scores)
                                    global_weighted_scores = weighted_scores
                                    prev_topic = user_topic
                                    number_qns_dict[prev_topic] = each_subject_asked
                                    subject_score = each_subject_correct/each_subject_asked
                                    each_subject_correct = 0
                                    each_subject_asked = 0
                                    score_tracking_dict[prev_topic] = subject_score
                                    # Create a message that includes the topics from suggested_roles
                                    topics_message = "Great! You are proficient in all topics. Suggested roles to apply for: "
                                    topics_message += ", ".join(suggested_roles)
                                    topics_message += ". Enter 'start test' to move to the exam module for certification!"
                                    #cursor.execute("DELETE FROM personalized_routes WHERE username = ?",(username))
                                    return render_template('chatbot.html', message=topics_message, topic=user_topic)
                                else:
                                    topics.remove(user_topic)
                                    global_route.remove(user_topic)
                                    session['topic'] = None
                                    session['progress'] = None
                                    cursor.execute("DELETE FROM user_progress WHERE username = ? AND topic = ?", (username, user_topic))
                                    new_route_str = ','.join(global_route)
                                    cursor.execute("UPDATE personalized_routes SET route = ? WHERE username = ?", (new_route_str, username))
                                    conn.commit()
                                    prev_topic = user_topic
                                    user_topic = topics[0]
                                    session['topic'] = user_topic
                                    user_progress_topic = user_progress[user_topic]
                                    user_progress_topic['current_question_index'] = -1
                                    number_qns_dict[prev_topic] = each_subject_asked
                                    subject_score = each_subject_correct/each_subject_asked
                                    each_subject_correct = 0
                                    each_subject_asked = 0
                                    score_tracking_dict[prev_topic] = subject_score
                                    return render_template('chatbot.html', message=f"You have completed {prev_topic}, and you had an accuracy of {subject_score:.2f}.. Let's move on to {user_topic}. Shall we start the quiz?", topic=user_topic)
                                
                            # Reset the question index for the new difficulty level
                            user_progress_topic['current_question_index'] = 0
                            current_question_index = user_progress_topic['current_question_index']
                            questions_of_current_difficulty = [
                                i for i, level in enumerate(difficulty_levels[user_topic]) if level == current_difficulty
                            ]
                            current_question = questions[user_topic][questions_of_current_difficulty[current_question_index]]
                            return render_template('chatbot.html', message=f"Great! Let's move on to {user_progress_topic['current_difficulty']} level questions. Here's your first question:\n{current_question}", topic=user_topic)
                    
                else:
                    # Check if there are more difficulty levels remaining
                    if current_difficulty == 'Easy':
                        if 'Medium' in difficulty_levels[user_topic]:
                            user_progress_topic['current_difficulty'] = 'Medium'
                            current_difficulty = user_progress_topic['current_difficulty']
                        elif 'Hard' in difficulty_levels[user_topic]:
                            user_progress_topic['current_difficulty'] = 'Hard'
                            current_difficulty = user_progress_topic['current_difficulty']
                    elif current_difficulty == 'Medium':
                        if 'Hard' in difficulty_levels[user_topic]:
                            user_progress_topic['current_difficulty'] = 'Hard'
                            current_difficulty = user_progress_topic['current_difficulty']
                    else:
                        print('Before check 2')
                        print(len(topics))
                        # Check if there are more topics remaining
                        if len(topics) == 1:
                            check = 1
                            weighted_scores = calculate_weighted_scores(score_tracking_dict, number_qns_dict, found_strong_topics,location_list)
                            suggested_roles = suggest_roles(weighted_scores)
                            global_weighted_scores = weighted_scores
                            prev_topic = user_topic
                            number_qns_dict[prev_topic] = each_subject_asked
                            subject_score = each_subject_correct/each_subject_asked
                            each_subject_correct = 0
                            each_subject_asked = 0
                            score_tracking_dict[prev_topic] = subject_score
                            # Create a message that includes the topics from suggested_roles
                            topics_message = "Great! You are proficient in all topics. Suggested roles to apply for: "
                            topics_message += ", ".join(suggested_roles)
                            topics_message += ". Enter 'start test' to move to the exam module for certification!"
                            #cursor.execute("DELETE FROM personalized_routes WHERE username = ?",(username))
                            return render_template('chatbot.html', message=topics_message, topic=user_topic)
                        else:
                            topics.remove(user_topic)
                            global_route.remove(user_topic)
                            session['topic'] = None
                            session['progress'] = None
                            cursor.execute("DELETE FROM user_progress WHERE username = ? AND topic = ?", (username, user_topic))
                            new_route_str = ','.join(global_route)
                            cursor.execute("UPDATE personalized_routes SET route = ? WHERE username = ?", (new_route_str, username))
                            conn.commit()
                            user_topic = topics[0]
                            session['topic'] = user_topic
                            user_progress_topic = user_progress[user_topic]
                            user_progress_topic['current_question_index'] = -1
                            return render_template('chatbot.html', message=f"Let's move on to {user_topic}. Shall we start the quiz?", topic=user_topic)

                    # Reset the question index for the new difficulty level
                    user_progress_topic['current_question_index'] = 0
                    current_question_index = user_progress_topic['current_question_index']
                    questions_of_current_difficulty = [
                        i for i, level in enumerate(difficulty_levels[user_topic]) if level == current_difficulty
                    ]
                    current_question = questions[user_topic][questions_of_current_difficulty[current_question_index]]
                    return render_template('chatbot.html', message=f"Great! Let's move on to {user_progress_topic['current_difficulty']} level questions. Here's your first question:\n{current_question}", topic=user_topic)

            # Initialize a response variable to store the bot's response
            response = None

            # Iterate through the intents
            for intent in data['intents']:
                if message.lower() in [pattern.lower() for pattern in intent['patterns']] or any(Levenshtein.distance(message.lower(), pattern.lower()) <= 1.5 for pattern in intent['patterns']):
                    response = random.choice(intent['responses'])
                    break  # Exit the loop once a match is found

            # Check if a response was found
            if response is not None:
                return render_template('chatbot.html', message=response, topic=user_topic)
            
            #if message.lower() in ['start test']:
            if any(Levenshtein.distance(message.lower(), ref) <= 1 for ref in ['start test','test']) or message.lower() in ['start test','test']:
                return redirect(url_for('test'))

            if user_progress_topic['current_question_index'] == -1:
                if message.lower() in [pattern.lower() for pattern in data['intents'][1]['patterns']]:
                    user_progress_topic['current_question_index'] = 0
                    current_question_index = user_progress_topic['current_question_index']
                    current_question = questions[user_topic][current_question_index]
                    return render_template('chatbot.html', message=f"Great! Let's start the {user_topic} quiz. Here's your first question:\n{current_question}", topic=user_topic)
                else:
                    return render_template('chatbot.html', message="I'm sorry, I didn't understand that. If you want to start a quiz, please let me know.", topic=user_topic)

            current_question_index = user_progress_topic['current_question_index']
            current_difficulty = user_progress_topic['current_difficulty']
            questions_of_current_difficulty = [
                i for i, level in enumerate(difficulty_levels[user_topic]) if level == current_difficulty
            ]
            current_question = questions[user_topic][questions_of_current_difficulty[current_question_index]]
            correct_answer = answers[user_topic][questions_of_current_difficulty[current_question_index]]
            explanation = explanations[user_topic][questions_of_current_difficulty[current_question_index]]

            if is_answer_similar(message, correct_answer, paraphrases):
                answer_score = get_similarity_score(message, correct_answer, paraphrases)
                if answer_score >= 0.8:
                    response = f"Correct! Your answer's score was {answer_score:.2f}. That was a great answer! Here's the explanation:\n{explanation}\n\nEnter 'okay' to continue."
                elif answer_score >= 0.6 and answer_score < 0.8:
                    response = f"Correct! Your answer's score was {answer_score:.2f}. Your answer was good and you can improve on it by mentioning a few more key terms related to the concept. Here's the explanation:\n{explanation}\n\nEnter 'okay' to continue."
                elif answer_score >= 0.42 and answer_score < 0.6:
                    response = f"Correct! Your answer's score was {answer_score:.2f}. Good attempt! However, you must improve the precision of your answer by giving a more detailed answer. Here's the explanation:\n{explanation}\n\nEnter 'okay' to continue."
                num_correct = num_correct + 1
                asked_questions = asked_questions + 1
                print(num_correct, " ", asked_questions)
            else:
                answer_score = get_similarity_score(message, correct_answer, paraphrases)
                if answer_score < 0.42 and answer_score >= 0.225:
                    response = f"Incorrect. The correct answer is '{correct_answer}'. Your answer's score was {answer_score:.2f}. You were close to the answer; however, here's an explanation to help you improve your answer: {explanation}\n\nEnter 'okay' to continue."
                else:
                    response = f"Incorrect. The correct answer is '{correct_answer}'. Your answer's score was {answer_score:.2f}. The answer can be improved significantly, and here's an explanation to help you improve your answer: {explanation}\n\nEnter 'okay' to continue."
                print(num_correct, " ", asked_questions)
                asked_questions = asked_questions + 1

            user_progress_topic['asked_questions'].append(current_question)

            # Use the loaded model for generating a response
            input_data = bag_of_words(message, words)
            input_data = np.array(input_data)
            result = model.predict([input_data])[0]
            # Assuming `labels` is the list of intent labels
            predicted_label = labels[np.argmax(result)]
            # Fetch the appropriate response based on the predicted label
            chatbot_response = get_response_from_label(predicted_label)

            session['progress'] = user_progress
            return render_template('chatbot.html', message=response, chatbot_response=chatbot_response, topic=user_topic)

    elif request.method == 'GET':
        if 'topic' not in session:
            for topic in topics:
                cursor.execute("SELECT current_question_index, current_difficulty FROM user_progress WHERE username = ? AND topic = ?", (username, topic))
                row = cursor.fetchone()
                if row is not None:
                    user_progress[topic] = {
                        'asked_questions': [],
                        'current_question_index': row[0],
                        'current_difficulty': row[1]
                    }
                else:
                    user_progress[topic] = {
                        'asked_questions': [],
                        'current_question_index': -1,
                        'current_difficulty': 'Easy'
                    }
            user_topic = topics[0]
            session['topic'] = user_topic
            session['progress'] = user_progress
            return render_template('chatbot.html', message=random.choice(data['intents'][0]['responses']), topic=user_topic)
        else:
            return render_template('chatbot.html', message=random.choice(data['intents'][0]['responses']), topic=session['topic'])

@app.route('/test', methods=['GET', 'POST'])
def test():
    global check,global_weighted_scores
    if 'username' not in session:
        return redirect(url_for('index'))

    global certified_topics, global_route, location_list
    username = session['username']
    # Fetch the personalized route topics
    cursor.execute("SELECT route FROM personalized_routes WHERE username = ?", (username,))
    row = cursor.fetchone()
    '''
    if row is not None:
        topics_text = row[0]
    else:
        topics_text = ""
    '''
    topics_text = ""

    topics = topics_text.split(', ')

    # Fetch the certified topics from the test_progress table
    cursor.execute("SELECT certified_topics FROM test_progress WHERE username = ?", (username,))
    row = cursor.fetchone()

    if row is not None and row[0]:
        certified_topics = row[0].split(',')
    else:
        certified_topics = []
    
    # Filter the global_route topics to get the topics for the test
    topics_for_test = [topic for topic in location_list if topic not in topics and topic not in certified_topics]
    print(topics_for_test)

    if not topics_for_test:
        #print(global_weighted_scores)
        # Get the current date and time
        #current_datetime = datetime.datetime.now()
        #proficiency_levels = classify_subjects(global_weighted_scores)
        # Render the certificate template, passing proficiency_levels
        #return render_template('certificate.html', username=username, certified_topics=certified_topics, completion_date=current_datetime, proficiency_levels=proficiency_levels)
        return redirect(url_for('user_dashboard'))

    if request.method == 'GET':
        return render_template('test.html', topics=topics_for_test)

    if request.method == 'POST':
        selected_topic = request.form['selected_topic']

        if selected_topic not in topics_for_test:
            return redirect(url_for('test'))
        
        print(selected_topic)

        # Assuming `selected_topic` is the topic chosen by the user
        questions_of_current_difficulty = [
            i for i, level in enumerate(difficulty_levels[selected_topic]) if level == 'Hard'
        ]
        # Get the questions for the selected topic and the current difficulty level
        selected_questions = [
            questions[selected_topic][index] for index in questions_of_current_difficulty
        ][:2]  # Select the first 10 questions

        # Store the selected topic in the session
        session['selected_topic'] = selected_topic

        # Store the selected questions and their correct answers in the session
        session['test_questions'] = selected_questions
        session['test_correct_answers'] = [answers[selected_topic][index] for index in questions_of_current_difficulty][:2]

        # Initialize the number of correct answers for this test
        session['test_num_correct'] = 0

        # Redirect the user to the first question
        return redirect(url_for('test_question', question_number=0))


@app.route('/test/question/<int:question_number>', methods=['GET', 'POST'])
def test_question(question_number):
    if 'username' not in session:
        return redirect(url_for('index'))

    global certified_topics, subj_correct_dict, subj_total_dict
    username = session['username']
    if 'test_questions' not in session or 'test_correct_answers' not in session:
        # If the user tries to access the question page directly without starting the test,
        # redirect them back to the test module
        return redirect(url_for('test'))

    selected_questions = session['test_questions']
    selected_correct_answers = session['test_correct_answers']

    if question_number >= len(selected_questions):
        # If all questions have been answered, show the results
        num_correct = session.get('test_num_correct', 0)
        num_total_questions = len(selected_questions)
        selected_topic = session.get('selected_topic')
        subj_total_dict[selected_topic] = num_total_questions
        subj_correct_dict[selected_topic] = num_correct

        # Check if the user has passed the test
        if num_correct >= 1:
            certified_topics.append(selected_topic)
            try:
                cursor.execute("""
                    MERGE INTO test_progress AS target
                    USING (SELECT ? AS username) AS source
                    ON target.username = source.username
                    WHEN MATCHED THEN
                        UPDATE SET target.certified_topics = ?
                    WHEN NOT MATCHED THEN
                        INSERT (username, certified_topics) VALUES (?, ?);
                """, (username, ','.join(certified_topics), username, ','.join(certified_topics)))
                conn.commit()

            except pyodbc.Error as e:
                # Handle any database-related errors here, e.g., log the error and show an error message to the user.
                print("Error updating certified topics in the database:", e)

        # Show the test results and ask the user what they want to do next
        return render_template('test_results.html', num_correct=num_correct, num_total=num_total_questions)

    if request.method == 'GET':
        # Display the current question to the user
        question = selected_questions[question_number]
        return render_template('test_question.html', question=question, question_number=question_number)

    if request.method == 'POST':
        # Check the user's response to the current question
        response = request.form.get('response', '').strip()
        correct_answer = selected_correct_answers[question_number]

        if is_answer_similar(response, correct_answer, paraphrases):
            # Increment the number of correct answers
            session['test_num_correct'] = session.get('test_num_correct', 0) + 1

        # Move to the next question
        return redirect(url_for('test_question', question_number=question_number + 1))


@app.route('/test/results', methods=['POST'])
def test_results():
    if 'username' not in session:
        return redirect(url_for('index'))

    global certified_topics
    # Get the user's response to the last question
    question_number = request.form.get('question_number', type=int)
    response = request.form.get('response', '').strip()
    selected_questions = session.get('test_questions', [])
    selected_correct_answers = session.get('test_correct_answers', [])
    selected_topic = session.get('selected_topic')
    username = session['username']

    if question_number is not None and 0 <= question_number < len(selected_questions):
        # Check the user's response to the last question
        correct_answer = selected_correct_answers[question_number]
        if is_answer_similar(response, correct_answer, paraphrases):
            # Increment the number of correct answers
            session['test_num_correct'] = session.get('test_num_correct', 0) + 1

    return redirect(url_for('test_question', question_number=question_number + 1))

@app.route('/dashboard')
def user_dashboard():
    global location_dict
    # Calculate the percentage scores based on training data
    percentage_scores = {}
    for topic in location_list:
        if topic in score_tracking_dict and topic in number_qns_dict:
            percentage_scores[topic] = score_tracking_dict[topic]* 100
        else:
            percentage_scores[topic] = 0

    # Calculate overall user performance statistics
    total_correct_answers = sum(subj_correct_dict.values())
    total_questions_answered = sum(subj_total_dict.values())
    overall_accuracy = (total_correct_answers / total_questions_answered) * 100

    # Include some basic statistical analysis here (e.g., average score, highest/lowest score, etc.)
    # Perform predictive analytics
    #predicted_scores = perform_predictive_analysis(score_tracking_dict, number_qns_dict)
    print(score_tracking_dict)
    print(number_qns_dict)
    print(subj_correct_dict)
    print(subj_total_dict)
    # Create a dictionary to store predicted values
    predicted_data = {}
    # Predict the next score for each topic and store in the dictionary
    for topic, scores in location_dict.items():
        next_score = predict_next_score(scores)
        percentage_score = (next_score / 2) * 100
        predicted_data[topic] = percentage_score

    # Create interactive plots using Plotly
    fig_score_tracking = create_plot('Score Tracking', score_tracking_dict)
    fig_number_qns = create_plot('Number of Questions', number_qns_dict)
    fig_subj_correct = create_plot('Subject Correct Answers', subj_correct_dict)
    fig_subj_total = create_plot('Subject Total Questions', subj_total_dict)

    # Calculate suggestions based on user's performance
    user_suggestions = calculate_suggestions(score_tracking_dict,number_qns_dict,subj_correct_dict,subj_total_dict)
    if not user_suggestions:
        user_suggestions.append(f"There are no suggestions")

    return render_template('dashboard.html', strong_topics=found_strong_topics, percentage_scores=percentage_scores, overall_accuracy=overall_accuracy, predicted_data=predicted_data,
                           fig_score_tracking=fig_score_tracking, fig_number_qns=fig_number_qns, fig_subj_correct=fig_subj_correct, fig_subj_total=fig_subj_total,user_suggestions=user_suggestions)

@app.route('/certificate')
def certificate():
    global global_weighted_scores
    username = session['username']
    current_datetime = datetime.datetime.now()
    global_weighted_scores['English Literature'] = 'Expert'
    print(global_weighted_scores)
    proficiency_levels = classify_subjects(global_weighted_scores)
    # Render the certificate template, passing proficiency_levels
    return render_template('certificate.html', username=username, certified_topics=certified_topics, completion_date=current_datetime, proficiency_levels=proficiency_levels)

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('progress', None)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=False)