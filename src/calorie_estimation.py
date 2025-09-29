"""
Calorie estimation for recognized food items
"""
import pandas as pd
import json
from config import config

class CalorieEstimator:
    def __init__(self):
        self.food_calories = self._load_calorie_data()
    
    def _load_calorie_data(self):
        """Load calorie data from CSV or create default"""
        try:
            # Try to load from CSV if exists
            df = pd.read_csv(config.NUTRITION_DATA_PATH)
            return dict(zip(df['food_name'], df['calories']))
        except:
            # Default calorie estimates for Food-101 classes
            return {
                'apple_pie': 265, 'baby_back_ribs': 430, 'baklava': 321,
                'beef_carpaccio': 184, 'beef_tartare': 195, 'beet_salad': 120,
                'beignets': 452, 'bibimbap': 430, 'bread_pudding': 321,
                'breakfast_burrito': 295, 'bruschetta': 184, 'caesar_salad': 184,
                'cannoli': 321, 'caprese_salad': 215, 'carrot_cake': 371,
                'ceviche': 158, 'cheesecake': 321, 'chicken_curry': 295,
                'chicken_wings': 203, 'chocolate_cake': 371, 'chocolate_mousse': 321,
                'churros': 452, 'clam_chowder': 215, 'club_sandwich': 350,
                'crab_cakes': 295, 'creme_brulee': 321, 'croque_madame': 350,
                'cup_cakes': 371, 'deviled_eggs': 158, 'donuts': 452,
                'dumplings': 265, 'edamame': 120, 'eggs_benedict': 350,
                'escargots': 184, 'falafel': 265, 'filet_mignon': 295,
                'fish_and_chips': 430, 'foie_gras': 452, 'french_fries': 312,
                'french_onion_soup': 215, 'french_toast': 350, 'fried_calamari': 312,
                'fried_rice': 430, 'frozen_yogurt': 207, 'garlic_bread': 312,
                'gnocchi': 265, 'greek_salad': 184, 'grilled_cheese_sandwich': 350,
                'grilled_salmon': 295, 'guacamole': 158, 'gyoza': 265,
                'hamburger': 295, 'hot_and_sour_soup': 120, 'hot_dog': 295,
                'huevos_rancheros': 350, 'hummus': 158, 'ice_cream': 207,
                'lasagna': 350, 'lobster_bisque': 215, 'lobster_roll_sandwich': 350,
                'macaroni_and_cheese': 350, 'macarons': 371, 'miso_soup': 120,
                'mussels': 158, 'nachos': 312, 'omelette': 265,
                'onion_rings': 312, 'oysters': 158, 'pad_thai': 430,
                'paella': 430, 'pancakes': 350, 'panna_cotta': 321,
                'peking_duck': 295, 'pho': 430, 'pizza': 266,
                'pork_chop': 295, 'poutine': 452, 'prime_rib': 430,
                'pulled_pork_sandwich': 350, 'ramen': 436, 'ravioli': 265,
                'red_velvet_cake': 371, 'risotto': 430, 'samosa': 265,
                'sashimi': 158, 'scallops': 158, 'seaweed_salad': 120,
                'shrimp_and_grits': 350, 'spaghetti_bolognese': 430,
                'spaghetti_carbonara': 430, 'spring_rolls': 265,
                'steak': 295, 'strawberry_shortcake': 321, 'sushi': 150,
                'tacos': 226, 'takoyaki': 265, 'tiramisu': 321,
                'tuna_tartare': 184, 'waffles': 291
            }
    
    def estimate_calories(self, food_class, confidence=1.0):
        """Estimate calories based on food class and confidence"""
        base_calories = self.food_calories.get(food_class, 250)
        
        # Adjust based on confidence (lower confidence = wider range)
        if confidence > 0.8:
            adjustment = 0.1  # ±10% for high confidence
        elif confidence > 0.6:
            adjustment = 0.2  # ±20% for medium confidence
        else:
            adjustment = 0.3  # ±30% for low confidence
        
        min_calories = int(base_calories * (1 - adjustment))
        max_calories = int(base_calories * (1 + adjustment))
        
        return {
            'estimated': base_calories,
            'range': f"{min_calories}-{max_calories}",
            'confidence': confidence
        }
    
    def get_nutrition_info(self, food_class, confidence=1.0):
        """Get complete nutrition information"""
        calorie_info = self.estimate_calories(food_class, confidence)
        
        # Sample nutrition breakdown (in reality, this would come from a database)
        nutrition_info = {
            'food_name': food_class.replace('_', ' ').title(),
            'calories': calorie_info,
            'protein': f"{(calorie_info['estimated'] * 0.15 / 4):.1f}g",
            'carbs': f"{(calorie_info['estimated'] * 0.55 / 4):.1f}g",
            'fat': f"{(calorie_info['estimated'] * 0.3 / 9):.1f}g",
            'confidence': confidence
        }
        
        return nutrition_info

# Test the calorie estimator
if __name__ == "__main__":
    estimator = CalorieEstimator()
    test_foods = ['pizza', 'sushi', 'hamburger', 'apple_pie']
    
    for food in test_foods:
        info = estimator.get_nutrition_info(food)
        print(f"{info['food_name']}: {info['calories']['estimated']} cal "
              f"({info['calories']['range']} cal range)")