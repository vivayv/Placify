USE capstone;

SELECT TABLE_NAME 
FROM capstone.INFORMATION_SCHEMA.TABLES 
WHERE TABLE_TYPE = 'BASE TABLE';

SELECT * FROM personalized_routes;
DELETE FROM personalized_routes WHERE username='Siddharth';

SELECT * FROM user_details;
DELETE FROM user_details WHERE username='Siddharth';

SELECT * FROM user_progress;
DELETE FROM user_progress WHERE username='Siddharth';

CREATE TABLE test_progress (
    id INT IDENTITY(1,1) PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    certified_topics VARCHAR(MAX)
);

SELECT * FROM test_progress;
DELETE FROM test_progress WHERE username='Siddharth';

SELECT * FROM Operating_System;
SELECT * FROM Object_Oriented_Programming;
SELECT * FROM Database_Management_System;
SELECT * FROM English_Literature;