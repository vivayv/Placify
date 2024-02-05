from flask import Flask, render_template
from flask import Flask, render_template, request, session, redirect, url_for
from sklearn.linear_model import LinearRegression
import plotly.express as px
import pandas as pd
import numpy as np
import datetime

app = Flask(__name__)

def classify_subjects(weighted_scores):
    proficiency_levels = {}
    
    for subject, score in weighted_scores.items():
        if score >= 40:
            proficiency_levels[subject] = 'Expert'
        elif 30 <= score < 40:
            proficiency_levels[subject] = 'Advanced'
        elif 20 <= score < 30:
            proficiency_levels[subject] = 'Intermediate'
        elif 10 <= score < 20:
            proficiency_levels[subject] = 'Beginner'
        else:
            proficiency_levels[subject] = 'Novice'
    
    return proficiency_levels

# Sample data (you can replace this with your actual data)
found_strong_topics = ['Topic1', 'Topic2', 'Topic3']
score_tracking_dict = {'Topic1': 80, 'Topic2': 75, 'Topic3': 90}
number_qns_dict = {'Topic1': 100, 'Topic2': 80, 'Topic3': 120}
subj_correct_dict = {'Topic1': 75, 'Topic2': 60, 'Topic3': 85}
subj_total_dict = {'Topic1': 100, 'Topic2': 80, 'Topic3': 120}

def perform_predictive_analysis(score_dict, question_dict):
    topics = found_strong_topics
    X = np.array([question_dict[topic] for topic in topics]).reshape(-1, 1)
    y = np.array([score_dict[topic] for topic in topics])

    # Create and fit a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Use the model to make predictions
    predicted_scores = model.predict(X)

    return predicted_scores

def create_plot(title, data_dict):
    data = {'Topic': list(data_dict.keys()), 'Value': list(data_dict.values())}
    df = pd.DataFrame(data)
    fig = px.bar(df, x='Topic', y='Value', title=title)
    return fig.to_html(full_html=False)

@app.route('/')
def index():
    return redirect(url_for('user_dashboard'))

@app.route('/dashboard')
def user_dashboard():
    # Calculate the percentage scores based on training data
    percentage_scores = {}
    for topic in found_strong_topics:
        if topic in score_tracking_dict and topic in number_qns_dict:
            percentage_scores[topic] = (score_tracking_dict[topic] / number_qns_dict[topic]) * 100
        else:
            percentage_scores[topic] = 0

    # Calculate overall user performance statistics
    total_correct_answers = sum(subj_correct_dict.values())
    total_questions_answered = sum(subj_total_dict.values())
    overall_accuracy = (total_correct_answers / total_questions_answered) * 100

    # Include some basic statistical analysis here (e.g., average score, highest/lowest score, etc.)
    # Perform predictive analytics
    predicted_data = {'Topic1': 80, 'Topic2': 75, 'Topic3': 90}

    # Create interactive plots using Plotly
    fig_score_tracking = create_plot('Score Tracking', score_tracking_dict)
    fig_number_qns = create_plot('Number of Questions', number_qns_dict)
    fig_subj_correct = create_plot('Subject Correct Answers', subj_correct_dict)
    fig_subj_total = create_plot('Subject Total Questions', subj_total_dict)

    return render_template('dashboard.html', strong_topics=found_strong_topics, percentage_scores=percentage_scores, overall_accuracy=overall_accuracy, predicted_data=predicted_data,
                           fig_score_tracking=fig_score_tracking, fig_number_qns=fig_number_qns, fig_subj_correct=fig_subj_correct, fig_subj_total=fig_subj_total)

@app.route('/certificate')
def certificate():
    username = 'test'
    certified_topics = ['Topic 1','Topic 2','Topic 3']
    global_weighted_scores = {'Topic 1':56,'Topic 2':45,'Topic 3':32}
    current_datetime = datetime.datetime.now()
    proficiency_levels = classify_subjects(global_weighted_scores)
    # Render the certificate template, passing proficiency_levels
    return render_template('certificate.html', username=username, certified_topics=certified_topics, completion_date=current_datetime, proficiency_levels=proficiency_levels)

if __name__ == '__main__':
    app.run(debug=True)

'''
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


def classify_subjects(weighted_scores):
    proficiency_levels = {}
    
    for subject, score in weighted_scores.items():
        if score >= 40:
            proficiency_levels[subject] = 'Expert'
        elif 30 <= score < 40:
            proficiency_levels[subject] = 'Advanced'
        elif 20 <= score < 30:
            proficiency_levels[subject] = 'Intermediate'
        elif 10 <= score < 20:
            proficiency_levels[subject] = 'Beginner'
        else:
            proficiency_levels[subject] = 'Novice'
    
    return proficiency_levels



def suggest_roles(weighted_scores, threshold=30):
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

    for role, required_subjects in roles.items():
        if all(subject in weighted_scores and weighted_scores[subject] >= threshold for subject in required_subjects):
            suggested_roles.append(role)

    return suggested_roles

location_list = ['Operating System', 'Object Oriented Programming', 'English Literature', 'Database Management System', 'Computer Networks', 'C', 'Aptitude', 'Data Structures']

# Sample scores for all subjects
score_tracking_dict = {
    'Operating System': 0.85,
    'Object Oriented Programming': 0.75,
    'English Literature': 0.60,
    'Database Management System': 0.90,
    'Computer Networks': 0.80,
    'C': 0.70,
    'Aptitude': 0.65,
    'Data Structures': 0.75,
}

# Sample numbers of questions answered for all subjects
number_qns_dict = {
    'Operating System': 50,
    'Object Oriented Programming': 45,
    'English Literature': 30,
    'Database Management System': 60,
    'Computer Networks': 55,
    'C': 40,
    'Aptitude': 35,
    'Data Structures': 48,
}

# Strong topics (unchanged)
found_strong_topics = [
    'Operating System',
    'Object Oriented Programming',
    'Database Management System',
    'Computer Networks',
]

weighted_scores = calculate_weighted_scores(score_tracking_dict, number_qns_dict, found_strong_topics,location_list)
suggested_roles = suggest_roles(weighted_scores)

print("Weighted Scores:", weighted_scores)
print("Suggested Roles:", suggested_roles)

# Get proficiency levels
proficiency_levels = classify_subjects(weighted_scores)
print(proficiency_levels)
'''