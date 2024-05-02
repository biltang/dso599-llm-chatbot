-- This file initializes the database with a table and data
CREATE TABLE IF NOT EXISTS dinosaurs (
    ID TEXT PRIMARY KEY,
    Name TEXT
);

INSERT INTO dinosaurs (ID, Name) VALUES ('T88', 'T-Rex');
INSERT INTO dinosaurs (ID, Name) VALUES ('V66', 'Velociraptor');