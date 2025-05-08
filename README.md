Data Preprocessing:

 •	Cleaned the dataset by replacing invalid ratings (e.g., -1) with NaN and dropping missing values.

•	Encoded unique user and anime IDs into numerical indices for compatibility with the machine learning model.

Model Design and Development:

•	Designed a neural network model using TensorFlow and Keras with separate embedding layers for users and anime.

•	Calculated the dot product of the flattened embedding vectors to predict user ratings.

Training and Evaluation:

•	Split the data into training and testing sets using scikit-learn's train_test_split.

•	Trained the model with the Adam optimizer and mean squared error (MSE) loss, and evaluated its performance using MSE and Root Mean Squared Error (RMSE).

Recommendation Generation:

•	Implemented a recommendation algorithm that filters out anime already rated by a user and predicts ratings for the remaining anime.

•	Extracted the top 10 anime with the highest predicted ratings as personalized recommendations. 
