"""
============================================================
  Indian House Price ML Prediction - Flask Web App
  CA Project | Python & Machine Learning Subject
============================================================
  Run:  python app.py
  Open: http://127.0.0.1:5000
============================================================
"""
import warnings; warnings.filterwarnings('ignore')
from flask import Flask, render_template, request, jsonify, send_from_directory
import pickle, json, numpy as np, os, traceback

app  = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

@app.template_filter('format_number')
def format_number(v):
    return f"{int(v):,}"

# Load model & metadata on startup
with open(os.path.join(BASE, 'model.pkl'), 'rb') as f:
    payload  = pickle.load(f)
model    = payload['model']
encoders = payload['encoders']
features = payload['features']

with open(os.path.join(BASE, 'metrics.json')) as f:
    metrics = json.load(f)

with open(os.path.join(BASE, 'state_city.json')) as f:
    state_city = json.load(f)


@app.route('/')
def index():
    return render_template('index.html', metrics=metrics,
                           state_city=json.dumps(state_city))


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(BASE, 'static'), filename)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        def enc(col, val):
            le = encoders[col]
            return int(le.transform([val])[0]) if val in le.classes_ else 0

        bhk          = int(data.get('bhk', 2))
        size         = int(data.get('size', 1000))
        year_built   = int(data.get('year_built', 2010))
        floor_no     = int(data.get('floor_no', 5))
        total_floors = int(data.get('total_floors', 10))
        amenities    = data.get('amenities', [])

        prop_age      = 2024 - year_built
        floor_ratio   = floor_no / max(total_floors, 1)
        amenity_count = len(amenities)
        has_pool      = 1 if 'Pool'   in amenities else 0
        has_gym       = 1 if 'Gym'    in amenities else 0
        has_garden    = 1 if 'Garden' in amenities else 0

        X = [[bhk, size, prop_age, floor_no, total_floors, floor_ratio,
              amenity_count, has_pool, has_gym, has_garden,
              enc('State', data.get('state','Delhi')),
              enc('City',  data.get('city','Delhi')),
              enc('Property_Type', data.get('property_type','Apartment')),
              enc('Furnished_Status', data.get('furnished','Semi-furnished')),
              enc('Public_Transport_Accessibility', data.get('transport','Medium')),
              enc('Parking_Space', data.get('parking','Yes')),
              enc('Facing', data.get('facing','North')),
              enc('Owner_Type', data.get('owner','Owner'))]]

        ml_pred = round(max(10.0, min(500.0, float(model.predict(X)[0]))), 2)

        # Rule-based estimate grounded in Indian market logic
        loc_mult   = {'Mumbai':2.8,'Delhi':2.2,'Bangalore':2.0,'Hyderabad':1.8,
                      'Pune':1.6,'Chennai':1.6,'Kolkata':1.3,'Ahmedabad':1.25,
                      'Surat':1.1,'Jaipur':1.1,'Chandigarh':1.3,'Coimbatore':1.1,
                      'Nagpur':1.0,'Lucknow':1.0,'Bhopal':0.95}
        type_mult  = {'Villa':1.45,'Independent House':1.2,'Apartment':1.0}
        furn_mult  = {'Furnished':1.12,'Semi-furnished':1.0,'Unfurnished':0.9}
        trans_mult = {'High':1.06,'Medium':1.0,'Low':0.94}
        face_mult  = {'South':1.04,'East':1.02,'North':1.0,'West':0.97}
        own_mult   = {'Builder':1.05,'Owner':1.0,'Broker':0.98}

        city = data.get('city','Delhi')
        estimated = (size * 4500
            * loc_mult.get(city, 1.0)
            * type_mult.get(data.get('property_type','Apartment'), 1.0)
            * furn_mult.get(data.get('furnished','Semi-furnished'), 1.0)
            * trans_mult.get(data.get('transport','Medium'), 1.0)
            * face_mult.get(data.get('facing','North'), 1.0)
            * own_mult.get(data.get('owner','Owner'), 1.0)
            * (1.03 if data.get('parking')=='Yes' else 1.0)
            * (1 + amenity_count * 0.03)
            * max(0.65, 1 - prop_age * 0.008)
            * (1 + (bhk - 2) * 0.08)
        ) / 100000

        return jsonify({'ml_prediction': ml_pred,
                        'rule_based':    round(max(10.0, min(2000.0, estimated)), 2),
                        'bhk': bhk, 'size': size, 'city': city,
                        'amenity_count': amenity_count})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 400

@app.route("/health")
def health():
    return "OK", 200

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  🏠  Indian House Price ML Dashboard")
    print("  📍  Open: http://127.0.0.1:5000")
    print("  ✅  Press CTRL+C to stop the server")
    print("="*55 + "\n")
    app.run(debug=True, port=5000, use_reloader=False)
