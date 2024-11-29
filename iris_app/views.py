import joblib
from django.shortcuts import render
from django.http import JsonResponse

# Load the trained model
model = joblib.load('iris_app/model/iris_model.pkl')

def home(request):
    return render(request, 'iris_app/home.html')  # Use the app-specific path

def predict(request):
    if request.method == 'POST':
        # Extract input features from the request
        try:
            sepal_length = float(request.POST['sepal_length'])
            sepal_width = float(request.POST['sepal_width'])
            petal_length = float(request.POST['petal_length'])
            petal_width = float(request.POST['petal_width'])
            features = [[sepal_length, sepal_width, petal_length, petal_width]]

            # Make prediction
            prediction = model.predict(features)
            species = ['Setosa', 'Versicolor', 'Virginica'][prediction[0]]
            
            return JsonResponse({'prediction': species})
        except Exception as e:
            return JsonResponse({'error': str(e)})
    return render(request, 'iris_app/predict.html')
from django.shortcuts import render

 
    
