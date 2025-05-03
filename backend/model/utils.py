import tensorflow as tf
import keras
import numpy as np
import random

months = ['January','February','March','April','May','June','July','August','September','October','November','December']
industries = ['Finance', 'Healthcare', 'Energy', 'Retail', 'Tech','Education']
countries = ['USA', 'UK', 'Germany', 'India', 'China', 'Russia', 'Brazil', 'France']

industry_risk = {
    "Finance": 0.9, "Energy": 0.8, "Retail": 0.6, "Tech": 0.7, "Education": 0.2
}


geopolitical_tension_dict = {
    "USA": 0.8,
    "UK": 0.2,
    "Germany": 0.5,
    "India": 0.7,
    "China":0.7,
    "Russia": 0.5,
    "Brazil": 0.2,
    "France": 0.3
}

country_risk = ["Russia", "China", "India"]

recent_attacks_dict = {
    "Finance": 5,
    "Energy": 2,
    "Retail": 4,
    "Tech": 4,
    "Education": 2
}

def populate_data_row():
    country = random.choice(countries)
    industry = random.choice(industries)
    cybersecurity_posture = np.random.uniform(0, 1)
    threat_intelligence = np.random.uniform(0, 1)
    is_high_risk_country = 1 if country in country_risk else 0
    industry_risk_score = industry_risk.get(industry, 0.5)
    recent_attacks = recent_attacks_dict.get(industry, 2) # Randomized lag feature
    geopolitical_tension = geopolitical_tension_dict.get(country, 2)  # External feature
    # cybersecurity of particular industry for a specific country
    # Base risk influenced by country and industry
    base_risk = 0.2 + 0.1 * is_high_risk_country  + 0.1 * industry_risk_score - 0.1 * cybersecurity_posture + 0.05 * recent_attacks + 0.1 * geopolitical_tension- 0.1 * threat_intelligence
    # Target: 1 if risk + noise > threshold
    risk_score = base_risk + np.random.normal(0, 0.1)
    label = int(risk_score > 0.5)

    return {
        'industry': industry,
        'country': country,
        'recent_attacks': recent_attacks,
        'geopolitical_tension': geopolitical_tension,
        'cybersecurity_posture': cybersecurity_posture,
        'threat_intelligence': threat_intelligence,
        'label': label
    }

def preprocess_input(data):
    recent_attacks = recent_attacks_dict.get(data.industry,2)
    geopolitical_tension = geopolitical_tension_dict.get(data.country, 2)
    return [industries.index(data.industry), countries.index(data.country),recent_attacks,geopolitical_tension, data.cybersecurity, data.threatIntelligence]

@keras.saving.register_keras_serializable(name="loss") # type: ignore
def weighted_bce(pos_weight=2.0):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weight = y_true * pos_weight + (1 - y_true)
        return tf.reduce_mean(weight * bce)
    return loss
