from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from flask_cors import CORS

app = Flask(__name__)
app.debug = True
CORS(app)  # Menambahkan CORS ke aplikasi Flask

# Load data
customer_interactions = pd.read_csv("customer_interactions.csv")
product_details = pd.read_csv("product_details.csv", delimiter=";")
purchase_history = pd.read_csv("purchase_history.csv", delimiter=";")

# Merge data
merged_data = pd.merge(customer_interactions, purchase_history, on="customer_id")

# Feature extraction
X = merged_data[['page_views', 'time_spent']]
y = merged_data['product_id']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model to disk
joblib.dump(model, 'model.pkl')

# Load the trained model from disk
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return 'This is the home page'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Check if 'customer_id' is present in the JSON data
        if 'customer_id' not in data:
            return jsonify({'error': 'Customer ID is required.'}), 400
        
        customer_id = int(data['customer_id'])
        
        # Check if the provided customer_id exists in the data
        if customer_id not in customer_interactions['customer_id'].values:
            return jsonify({'error': 'Invalid customer ID.'}), 400
        
        page_views = float(data['page_views'])
        time_spent = float(data['time_spent'])
        
        # Prediction
        prediction = model.predict([[page_views, time_spent]])

        # Get product information
        product_info = product_details[product_details['product_id'] == prediction[0]]

        predicted_product = {
            'product_id': int(prediction[0]),
            'category': product_info['category'].values[0],
            'price': float(product_info['price'].values[0]),
            'ratings': float(product_info['ratings'].values[0])
        }

        # Print the JSON response
        print("Predicted Product:", predicted_product)

        # Return prediction result as JSON
        return jsonify(predicted_product)
    
    except KeyError as e:
        return jsonify({'error': f'Missing key in JSON data: {str(e)}'}), 400
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
