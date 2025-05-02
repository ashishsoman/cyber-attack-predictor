import tensorflow as tf
import keras
import numpy as np

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

def preprocess_input(data):
    recent_attacks = recent_attacks_dict.get(data.industry,2)
    geopolitical_tension = geopolitical_tension_dict.get(data.country, 2)
    return [industries.index(data.industry), countries.index(data.country),recent_attacks,geopolitical_tension, data.cybersecurity]

@keras.saving.register_keras_serializable(name="loss") # type: ignore
def weighted_bce(pos_weight=2.0):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weight = y_true * pos_weight + (1 - y_true)
        return tf.reduce_mean(weight * bce)
    return loss
