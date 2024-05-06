from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
import joblib

class MLModel(APIView):
    def get(self, request, input):
        input_data = input
        print('input_data12aaaaaaaaaaaaaaaaaaaaaaaaaaa' , input_data)
        # Assuming you receive the new_data as a JSON object in the request
        # input_data = request.data.get('input', '')
        if input_data:
            # Load the trained model
            classifier = joblib.load('trained_model_final1.pkl')

            # Load the vectorizer
            vectorizer = joblib.load('tfidf_vectorizer_final1.pkl')

            # Example prediction
            new_input_vectorized = vectorizer.transform([input_data])
            predicted_output = classifier.predict(new_input_vectorized)
            print("Predicted output:", predicted_output[0])
            return Response({"output": predicted_output[0]}, status=status.HTTP_200_OK)

        else:
            return Response({"error": "Please provide a valid 'input' string in the request."}, status=status.HTTP_400_BAD_REQUEST)




