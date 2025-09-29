import sys
sys.path.append('src')

from calorie_estimation import CalorieEstimator

print('🧪 Testing Calorie Estimation:')
estimator = CalorieEstimator()
test_foods = ['pizza', 'sushi', 'hamburger', 'apple_pie']

for food in test_foods:
    info = estimator.get_nutrition_info(food)
    print(f"{info['food_name']}: {info['calories']['estimated']} calories")