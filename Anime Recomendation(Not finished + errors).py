#Environment Setup
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Embedding, Flatten, Dot, Dense
from tensorflow.python.keras.models import Model
from sklearn.model_selection import train_test_split

#Load and Preprocess the Anime Recommendations Database
ratings = pd.read_csv('rating.csv')
ratings.replace({-1: np.nan}, inplace = True)
ratings.dropna(inplace = True)

#Encode Users and Anime IDs
# 1.Extracting Unique User and Anime IDs (Extracting unique user and anime IDs)
user_ids = ratings["user_id"].unique().tolist()
anime_ids = ratings["anime_id"].unique().tolist()
# 2.Creating Encoding Dictionaries (Creating dictionaries to encode user and anime IDs to numerical indices)
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
anime2anime_encoded = {x: i for i, x in enumerate(anime_ids)}
anime_encoded2anime = {i: x for i, x in enumerate(anime_ids)}
# 3. Encoding User and Anime IDs in the DataFrame (Replacing the original user and anime IDs in the DataFrame with their corresponding numerical indices)
ratings["user"] = ratings["user_id"].map(user2user_encoded)
ratings["anime"] = ratings["anime_id"].map(anime2anime_encoded)

# Split the Data into Training and Testing Sets
# 1. Getting Number of Users and Animes
num_users = len(user2user_encoded)
num_animes = len(anime_encoded2anime)
# 2. Converting Ratings to Float32
ratings["rating"] = ratings["rating"].values.astype(np.float32)
# 3. Extracting Features and Targets
X = ratings[["user", "anime"]].values
y = ratings["rating"].values
# 4. Splitting Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Model
# 1.Setting Embedding Size
embedding_size = 50
# 2.Creating Input Layers
user_input = Input(shape=(1,), name="user_input")
anime_input = Input(shape=(1,), name="anime_input")
# 3.Embedding Users and Animes
user_embedding = Embedding(num_users, embedding_size, name="user_embedding")(user_input)
anime_embedding = Embedding(num_animes, embedding_size, name="anime_embedding")(anime_input)
# 4.Flattening Embeddings
user_vec = Flatten(name="flatten_users")(user_embedding)
anime_vec = Flatten(name="flatten_animes")(anime_embedding)
# 5.Calculating Dot Product
dot_product = Dot(name="dot_product", axes=1)([user_vec, anime_vec])
# 6.Creating the Model
model = Model(inputs=[user_input, anime_input], outputs=dot_product)

# Compile and Train the Model
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit([X_train[:, 0], X_train[:, 1]], y_train, batch_size=64, epochs=5, verbose=1, validation_data=([X_test[:, 0], X_test[:, 1]], y_test))

# 7. Making Recommendations
user_id = user_ids[0]
user_enc = user2user_encoded[user_id]
user_anime_ids = ratings[ratings["user_id"]==user_id]["anime_id"].values
user_anime_ids = [anime2anime_encoded[x] for x in user_anime_ids]
all_anime_ids = list(set(range(num_animes)) - set(user_anime_ids))
user_encs = np.array([user_enc] * len(all_anime_ids))
ratings_pred = model.predict([user_encs, all_anime_ids])
top_10_indices = ratings_pred.flatten().argsort()[-10:][::-1]
recommended_anime_ids = [anime_encoded2anime[x] for x in top_10_indices]
print("Recommended anime ids:", recommended_anime_ids)

# 8.Evaluate the Model
# Make predictions on the test set
y_pred = model.predict([X_test[:, 0], X_test[:, 1]])

# Compute the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)

# Compute the root mean squared error
rmse = np.sqrt(mse)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

