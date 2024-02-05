DELETE FROM user_details where username = 'Siddharth';
DELETE FROM personalized_routes where username = 'Siddharth';

SELECT * FROM personalized_routes;

SELECT * FROM user_details;

CREATE TABLE user_progress (
  username VARCHAR(255) NOT NULL,
  topic VARCHAR(255) NOT NULL,
  current_question_index INT NOT NULL,
  current_difficulty VARCHAR(255) NOT NULL,
  PRIMARY KEY (username, topic)
);

SELECT * FROM user_progress;

SELECT * FROM personalized_routes;