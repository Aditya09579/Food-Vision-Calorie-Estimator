import sys
sys.path.append('src')
from predict import FoodPredictor
import os

def main():
    print('Testing Food Predictor...')
    predictor = FoodPredictor()
    if not os.path.exists('models/best_food_model.h5'):
        print('Model not found!')
        return
    print('Predictor ready!')
    print(f'Can recognize {len(predictor.class_names)} food classes')

if __name__ == '__main__':
    main()
