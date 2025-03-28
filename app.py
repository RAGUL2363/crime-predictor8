from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file
import pandas as pd
import pickle
import os
from io import StringIO, BytesIO
import csv

# Use a non-interactive backend to avoid Tkinter errors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import folium

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# ---------------------------------------------------------------------
# Load Trained Model (If you're doing Crime Predictions)
# ---------------------------------------------------------------------
model = pickle.load(open('models/model.pkl', 'rb'))

# ---------------------------------------------------------------------
# Load Dataset and Clean Columns
# ---------------------------------------------------------------------
csv_path = "data/crime_data.csv"
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

df.rename(columns={
    'Year': 'year',
    'District': 'district',
    'Crime Category': 'crime_category',
    'Crime Severity': 'crime_severity',
    'Number of Cases': 'num_cases'
}, inplace=True)

# ---------------------------------------------------------------------
# District Coordinates for Tamil Nadu (All 38 Districts)
# ---------------------------------------------------------------------
district_coords = {
    "Ariyalur": [11.1598, 79.1340],
    "Chengalpattu": [12.6916, 79.9873],
    "Chennai": [13.0827, 80.2707],
    "Coimbatore": [11.0168, 76.9558],
    "Cuddalore": [11.7473, 79.7683],
    "Dharmapuri": [12.1264, 78.1643],
    "Dindigul": [10.3644, 77.9176],
    "Erode": [11.3466, 77.7272],
    "Kallakurichi": [11.7760, 79.0741],
    "Kancheepuram": [12.8333, 79.7000],
    "Kanniyakumari": [8.0883, 77.5385],
    "Karur": [10.9600, 78.0760],
    "Krishnagiri": [12.5261, 78.2144],
    "Madurai": [9.9252, 78.1198],
    "Mayiladuthurai": [11.1032, 79.6525],
    "Nagapattinam": [10.7655, 79.8413],
    "Namakkal": [11.2333, 78.1667],
    "Nilgiris": [11.4074, 76.6955],
    "Perambalur": [11.2500, 78.8500],
    "Pudukkottai": [10.3792, 78.8242],
    "Ramanathapuram": [9.3718, 78.8371],
    "Ranipet": [12.7393, 79.3120],
    "Salem": [11.6643, 78.1460],
    "Sivagangai": [9.8807, 78.6672],
    "Tenkasi": [8.9705, 77.3161],
    "Thanjavur": [10.7870, 79.1378],
    "Theni": [9.9402, 77.5137],
    "Thoothukudi": [8.7642, 78.1348],
    "Tiruchirappalli": [10.7905, 78.7047],
    "Tirunelveli": [8.7139, 77.7566],
    "Tiruppur": [11.1085, 77.3411],
    "Tiruvallur": [13.1500, 79.9167],
    "Tiruvannamalai": [12.2250, 79.0700],
    "Tiruvarur": [10.7725, 79.6368],
    "Tirupattur": [12.4960, 78.2132],
    "Vellore": [12.9167, 79.1333],
    "Viluppuram": [11.9500, 79.4833],
    "Virudhunagar": [9.5760, 77.9000]
}

# ---------------------------------------------------------------------
# Simple Classification for Crime Rate
# ---------------------------------------------------------------------
def classify_crime_rate(rate):
    if rate < 10:
        return "Low"
    elif rate < 30:
        return "Medium"
    else:
        return "High"

# ---------------------------------------------------------------------
# Rule-Based Chatbot (Local FAQ)
# ---------------------------------------------------------------------
FAQ = {
    "how do i select the year?": "You can choose the year from the dropdown labeled 'Year' at the top of the form.",
    "what is this app for?": "This application predicts crime statistics based on your selected inputs like year, district, crime category, and severity.",
    "how do i select a district?": "Select your district from the dropdown; it contains all districts available in Tamil Nadu.",
    "what is crime category?": "Crime category refers to the type of crime such as Theft, Murder, or Drug Offenses.",
    "how does crime severity affect predictions?": "Crime severity helps determine the seriousness of a crime, which influences the predicted number of cases.",
    "how to export results?": "On the results page, you can click on 'Export Result as CSV' to download your prediction report.",
    "what does crime rate mean?": "Crime rate is the number of cases per 100,000 people, indicating how prevalent a crime is in a given area.",
    "how accurate is the prediction?": "The model's test accuracy is displayed during training. Accuracy depends on the quality of historical data.",
    "can i see analytics?": "Yes, the results page includes various analytics such as trend charts, top crime categories, heatmaps, and more.",
    "what if my question is not listed?": "If your question isn't recognized, please try rephrasing it or ask another question.",
    "how does the chatbot work?": "Our chatbot uses a rule-based approach to match your input to a set of predefined questions and returns the corresponding answer.",
    "can i get help on using the app?": "Absolutely, feel free to ask any question related to the app's features, such as how to select options or export data."
}

def get_rule_based_response(user_input):
    user_input_lower = user_input.lower()
    for question, answer in FAQ.items():
        if question in user_input_lower:
            return answer
    return "I'm not sure how to answer that. Please try rephrasing your question."

# ---------------------------------------------------------------------
# Home Route: Show Input Form
# ---------------------------------------------------------------------
@app.route('/')
def home():
    unique_years = sorted(df['year'].unique())
    unique_districts = sorted(df['district'].unique())
    unique_crime_categories = sorted(df['crime_category'].unique())
    unique_crime_severity = sorted(df['crime_severity'].unique())
    return render_template('index.html',
                           years=unique_years,
                           districts=unique_districts,
                           crime_categories=unique_crime_categories,
                           crime_severity=unique_crime_severity)

# ---------------------------------------------------------------------
# Predict Route: Process User Inputs, Generate Charts/Heatmap
# ---------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    year = int(request.form['year'])
    district = request.form['district']
    crime_category = request.form['crime_category']
    crime_severity = request.form['crime_severity']

    # Prepare input for ML model
    input_df = pd.DataFrame({
        'year': [year],
        'district': [district],
        'crime_category': [crime_category],
        'crime_severity': [crime_severity]
    })
    input_df = pd.get_dummies(input_df, columns=['district','crime_category','crime_severity'], drop_first=True)

    # Ensure columns match trained model
    trained_columns = model.feature_names_in_
    for col in trained_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[trained_columns]

    # Predict number of cases
    predicted_cases = model.predict(input_df)[0]

    # Retrieve additional info from dataset
    filtered_data = df[
        (df['year'] == year) &
        (df['district'] == district) &
        (df['crime_category'] == crime_category) &
        (df['crime_severity'] == crime_severity)
    ]
    if filtered_data.empty:
        additional = {
            "crime_rate": "Data Not Available",
            "crime_location": "Unknown",
            "festival_event": "None"
        }
    else:
        additional = {
            "crime_rate": round(filtered_data['Crime Rate per 100,000'].mean(), 2) if 'Crime Rate per 100,000' in filtered_data.columns else "N/A",
            "crime_location": ", ".join(filtered_data['Crime Location'].dropna().unique()),
            "festival_event": ", ".join(filtered_data['Festival/Event'].dropna().unique())
        }

    # Classify crime rate
    try:
        cr = float(additional["crime_rate"])
        crime_rate_class = classify_crime_rate(cr)
    except:
        crime_rate_class = "Unknown"

    # Store results in session
    session['selected'] = {
        'year': year,
        'district': district,
        'crime_category': crime_category,
        'crime_severity': crime_severity,
        'predicted_cases': round(predicted_cases),
        'crime_rate': additional["crime_rate"],
        'crime_rate_class': crime_rate_class,
        'crime_location': additional["crime_location"],
        'festival_event': additional["festival_event"]
    }

    # Generate analytics
    generate_trend_chart(district)
    generate_top_categories_chart(district)
    generate_top5_chart(district)
    generate_breakdown_chart(district)
    generate_heatmap(district)

    return redirect(url_for('results'))

# ---------------------------------------------------------------------
# Results Route: Display Prediction + Analytics
# ---------------------------------------------------------------------
@app.route('/results')
def results():
    selected = session.get('selected', {})
    return render_template('results.html', selected=selected)

# ---------------------------------------------------------------------
# Export CSV Route (Fixed with StringIO)
# ---------------------------------------------------------------------
@app.route('/export_csv')
def export_csv():
    selected = session.get('selected', {})
    output = StringIO()
    writer = csv.writer(output)

    # Write CSV headers
    writer.writerow([
        "Year", "District", "Crime Category", "Crime Severity",
        "Predicted Cases", "Crime Rate", "Crime Rate Category",
        "Crime Location", "Festival/Event"
    ])

    # Write CSV row
    writer.writerow([
        selected.get('year'),
        selected.get('district'),
        selected.get('crime_category'),
        selected.get('crime_severity'),
        selected.get('predicted_cases'),
        selected.get('crime_rate'),
        selected.get('crime_rate_class'),
        selected.get('crime_location'),
        selected.get('festival_event')
    ])

    # Convert text to bytes
    data = output.getvalue().encode('utf-8')
    output.close()

    return send_file(
        BytesIO(data),
        mimetype="text/csv",
        as_attachment=True,
        attachment_filename="crime_prediction.csv"
    )

# ---------------------------------------------------------------------
# Chatbot Route (Rule-Based)
# ---------------------------------------------------------------------
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"response": "Please ask a question."})

    response_text = get_rule_based_response(user_message)
    return jsonify({"response": response_text})

# ------------------------ Analytics Functions -------------------------
def generate_trend_chart(district):
    trend_df = df[df['district'] == district].groupby('year')['num_cases'].mean().reset_index()
    plt.figure(figsize=(6,4))
    plt.plot(trend_df['year'], trend_df['num_cases'], marker='o', color='blue')
    plt.xlabel('Year')
    plt.ylabel('Avg. Number of Cases')
    plt.title(f"Crime Trend in {district}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("static/trend.png")
    plt.close()

def generate_top_categories_chart(district):
    top_df = df[df['district'] == district].groupby('crime_category')['num_cases'].sum().reset_index()
    top_df = top_df.sort_values('num_cases', ascending=False)
    plt.figure(figsize=(6,4))
    plt.bar(top_df['crime_category'], top_df['num_cases'], color='skyblue')
    plt.xlabel('Crime Category')
    plt.ylabel('Total Cases')
    plt.title(f"Top Crime Categories in {district}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/top_categories.png")
    plt.close()

def generate_top5_chart(district):
    top5_df = df[df['district'] == district].groupby('crime_category')['num_cases'].sum().reset_index()
    top5_df = top5_df.sort_values('num_cases', ascending=False).head(5)
    plt.figure(figsize=(6,4))
    plt.bar(top5_df['crime_category'], top5_df['num_cases'], color='salmon')
    plt.xlabel('Crime Category')
    plt.ylabel('Total Cases')
    plt.title(f"Top 5 High Crimes in {district}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/top5.png")
    plt.close()

def generate_breakdown_chart(district):
    breakdown_df = df[df['district'] == district].groupby('crime_severity')['num_cases'].sum().reset_index()
    plt.figure(figsize=(6,4))
    plt.bar(breakdown_df['crime_severity'], breakdown_df['num_cases'], color='limegreen')
    plt.xlabel('Crime Severity')
    plt.ylabel('Total Cases')
    plt.title(f"Crime Breakdown in {district}")
    plt.tight_layout()
    plt.savefig("static/breakdown.png")
    plt.close()

def generate_heatmap(district):
    coords = district_coords.get(district, [10.0, 78.0])
    m = folium.Map(location=coords, zoom_start=10)
    folium.CircleMarker(location=coords, radius=20, popup=f"{district}", color="red", fill=True).add_to(m)
    m.save("static/heatmap.html")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
