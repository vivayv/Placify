import numpy as np
import random

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
    self.num_states = len(location_to_state)
    self.state_to_location = state_to_location

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
    global_route.extend([x for x in route if x not in global_route])
    #print(global_route)

global_route = []
qagent = QAgent(alpha,gamma,location_to_state,actions,rewards,state_to_location)

strong_topics = ['Operating System']
weak_topics = ['Aptitude']

if len(strong_topics) > 0 and len(weak_topics) > 0:
    # Choose a strong topic and a weak topic
    strong_topic = random.sample(strong_topics, 1)
    weak_topic = random.sample(weak_topics, 1)
    strong_topic = strong_topic[0]
    weak_topic = weak_topic[0]

    global_route = []

    # Call qagent.training with the chosen elements
    qagent.training(weak_topic, strong_topic, 1000)
    print("After succesfully running qlearning: ")
    print(global_route)

    # Continue to connect the topics until all topics are in the route
    missing_topics = [topic for topic in location_list if topic != 'English Literature']
    for topic in missing_topics:
        if topic not in global_route:
            global_route.append(topic)

print("If this runs then its entering and exiting the previous block properly")
print(global_route)