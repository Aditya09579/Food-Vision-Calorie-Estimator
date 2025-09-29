import sys
sys.path.append('src')
from predict import FoodPredictor
import os
import glob

def main():
    print('Testing with actual food images...')
    predictor = FoodPredictor()
    pizza_path = 'data/food-101/food-101/images/pizza'
    if os.path.exists(pizza_path):
        pizza_images = glob.glob(os.path.join(pizza_path, '*.jpg'))[:3]
        if pizza_images:
            print(f'Found {len(pizza_images)} pizza images for testing')
            for img_path in pizza_images:
                print('='*60)
                predictor.predict(img_path, top_k=3)
        else:
            print('No pizza images found')
    else:
        print('Pizza folder not found')

if __name__ == '__main__':
    main()
