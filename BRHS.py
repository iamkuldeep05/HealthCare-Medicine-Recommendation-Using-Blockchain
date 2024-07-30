from flask import Flask, jsonify, request
import hashlib
import json
from time import time
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import random

app = Flask(__name__)

# Load the data
data = pd.read_csv("healthcare.csv")

# Drop rows with NaN values
data.dropna(inplace=True)

# Split features and target variable
X = data.drop(columns=["Medicine", "Disease"])
y = data["Medicine"]

# Define preprocessing for numerical features
numerical_features = ['YearOfBirth', 'MonthOfBirth', 'DayOfBirth']

# Define preprocessing for categorical features
categorical_features = ['Gender', 'Symptoms', 'Causes', 'Name']

# Function to extract year, month, and day from date string
def extract_date_features(df):
    df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], errors='coerce', format='%d-%m-%Y')
    df['YearOfBirth'] = df['DateOfBirth'].dt.year
    df['MonthOfBirth'] = df['DateOfBirth'].dt.month
    df['DayOfBirth'] = df['DateOfBirth'].dt.day
    return df.drop(columns=['DateOfBirth'])

# Define preprocessing for numerical features
numerical_transformer = SimpleImputer(strategy='constant')

# Define preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Preprocess the data
X_preprocessed = extract_date_features(X)

# Train the model
model.fit(X_preprocessed, y)

class Block:
    def __init__(self, index, timestamp, patient_data, previous_hash, medicine, nonce=0):
        self.index = index
        self.timestamp = timestamp
        self.patient_data = patient_data
        self.previous_hash = previous_hash
        self.medicine = medicine
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "patient_data": self.patient_data,
            "previous_hash": self.previous_hash,
            "medicine": self.medicine,
            "nonce": self.nonce
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def mine_block(self, difficulty):
        while self.hash[:difficulty] != '0' * difficulty:
            self.nonce += 1
            self.hash = self.calculate_hash()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.difficulty = 4  # Adjust difficulty as needed

    def create_genesis_block(self):
        return Block(0, time(), "Genesis Block", "0", "None")

    def add_block(self, block):
        block.previous_hash = self.get_previous_hash()  # Set previous_hash to hash of the last block
        block.mine_block(self.difficulty)
        self.chain.append(block)

    def get_previous_hash(self):
        return self.chain[-1].hash if len(self.chain) > 0 else "0"

blockchain = Blockchain()

def generate_random_patient_data():
    genders = ['Male', 'Female']
    symptoms = ['Fever', 'Cough', 'Headache', 'Fatigue', 'Shortness of breath']
    causes = ['Viral Infection', 'Stress', 'Pollution']
    names = ['John Doe', 'Jane Smith', 'Michael Lee', 'Maria Garcia', 'David Johnson']
    
    random_data = {
        'Name': random.choice(names),
        'DateOfBirth': f"{random.randint(1, 28)}-{random.randint(1, 12)}-{random.randint(1950, 2005)}",
        'Gender': random.choice(genders),
        'Symptoms': ', '.join(random.sample(symptoms, random.randint(1, len(symptoms)))),
        'Causes': random.choice(causes)
    }
    return random_data


@app.route('/mine_blocks', methods=['POST'])
def mine_blocks():
    patient_data = generate_random_patient_data()
    index = len(blockchain.chain)
    timestamp = time()
    new_patient = extract_date_features(pd.DataFrame([patient_data]))  # Wrap patient_data in a list
    predicted_medicine = model.predict(new_patient)[0]
    
    new_block = Block(index, timestamp, new_patient.to_dict(orient='records'), blockchain.get_previous_hash(), predicted_medicine)
    blockchain.add_block(new_block)
    
    response = {
        'message': "Congratulations! You just mined a block!",
        'block_index': new_block.index,
        'timestamp': new_block.timestamp,
        'patient_data': new_block.patient_data,
        'medicine': new_block.medicine,
        'nonce': new_block.nonce,
        'hash': new_block.hash,
        'previous_hash': new_block.previous_hash
    }
    return jsonify(response), 200

@app.route('/get_chain', methods=['GET'])
def get_chain():
    chain_data = []
    for block in blockchain.chain:
        chain_data.append({
            'index': block.index,
            'timestamp': block.timestamp,
            'patient_data': block.patient_data,
            'medicine': block.medicine,
            'nonce': block.nonce,
            'hash': block.hash,
            'previous_hash': block.previous_hash
        })
    return jsonify({'chain': chain_data}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
