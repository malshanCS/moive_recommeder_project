# Movie Recommendation System

This project implements a movie recommendation system using machine learning techniques such as cosine similarity and TF-IDF vectorization.

## Overview

The movie recommendation system suggests similar movies based on the input movie name provided by the user. It uses cosine similarity to measure the similarity between movies and TF-IDF vectorization to represent movies as numerical feature vectors.

## Features

- **Movie Search**: Users can enter the name of a movie and get recommendations based on their input.
- **Best Match Suggestions**: If an exact match is not found, the system provides a list of best match suggestions for the user to choose from.
- **Top 10 Recommendations**: The system recommends the top 10 movies that are most similar to the selected movie.

## Technologies Used

- **Python**: The core programming language used for implementing the recommendation system.
- **FastAPI**: A modern, fast, web framework for building APIs with Python.
- **cosine_similarity**: A mathematical measure used to determine the similarity between movies.
- **TfidfVectorizer**: A feature extraction technique that converts movie descriptions into numerical feature vectors.

## Usage

1. Install the required dependencies by running the command `pip install -r requirements.txt`.
2. Run the FastAPI app by executing the command `python app.py`.
3. Open a web browser and navigate to `http://localhost:5000/recommendations`.
4. Enter the name of a movie in the input field and submit the form.
5. View the recommended movies based on the input movie.

## Dataset

The recommendation system uses a dataset of movies, along with their descriptions and other relevant information. The dataset is not included in this repository, but you can provide your own dataset to train and test the recommendation system.

## Acknowledgements

We would like to acknowledge the following resources that were instrumental in the development of this project:

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Python.org](https://www.python.org/)

