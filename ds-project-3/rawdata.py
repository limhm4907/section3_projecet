import os
import csv
import sqlite3

# CSV,DB File Path
CSV_FILEPATH = os.path.join(os.path.dirname(__file__), 'Telco_Customer_Churn.csv')
DB_FILEPATH = os.path.join(os.getcwd(), 'Churn.db')

# connect database
conn = sqlite3.connect(DB_FILEPATH)
cur = conn.cursor()

# create database table
cur.execute("DROP TABLE IF EXISTS Customer;")  # table name: Customer
cur.execute("""CREATE TABLE Customer (
    id VARCHAR(20) NOT NULL PRIMARY KEY,
    gender VARCHAR(10) NOT NULL,
    senior INTEGER NOT NULL,
    partner VARCHAR(10) NOT NULL,
    dependents VARCHAR(10) NOT NULL,
    tenure INTEGER NOT NULL,
    phone VARCHAR(10) NOT NULL,
    multiplelines VARCHAR(20) NOT NULL,
    internet VARCHAR(10) NOT NULL,
    onlinesecurity VARCHAR(10) NOT NULL,
    onlinebackup VARCHAR(10) NOT NULL,
    deviceprotection VARCHAR(10) NOT NULL,
    techsupport VARCHAR(10) NOT NULL,
    streamingtv VARCHAR(10) NOT NULL,
    streamingmovies VARCHAR(10) NOT NULL,
    contract VARCHAR(20) NOT NULL,
    paperlessbilling VARCHAR(10) NOT NULL,
    paymentmethod VARCHAR(30) NOT NULL,
    monthlycharges FLOAT NOT NULL,
    totalcharges FLOAT NOT NULL,
    churn VARCHAR(10) NOT NULL);""")

# insert into database table
with open(CSV_FILEPATH, 'r', newline='') as f:
    reader = csv.reader(f)
    data = [i for i in reader]
    data_list = data[1:]
for row in data_list:
    cur.execute("""INSERT INTO Customer (id, gender, senior, partner, dependents, tenure,
       phone, multiplelines, internet, onlinesecurity,
       onlineBackup, deviceprotection, techsupport, streamingtv,
       streamingmovies, contract, paperlessbilling, paymentmethod,
       monthlycharges, totalcharges, churn) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", tuple(row))

conn.commit()
conn.close()