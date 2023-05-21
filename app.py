import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('../data/movies.csv')
selected_features = ['genres', 'director', 'cast', 'keywords','budget','revenue','runtime','tagline']

# get the null values
for col in selected_features:
    # print(col, data[col].isnull().sum())
    data[col] = data[col].fillna('') # fill the null values with empty string


# combine all the features
movie_cols = data['genres'] + ' ' + data['director'] + ' ' + data['cast'] + ' ' + data['keywords'] + ' ' + data['tagline']

# converting txt data to features using TD-IDF vectorizer
vectorizer = TfidfVectorizer()
movie_features = vectorizer.fit_transform(movie_cols)


# find the similarity confidence value between the movies
similarity_confidence = cosine_similarity(movie_features)

similarity_confidence_df = pd.DataFrame(similarity_confidence)


# create a list of all the movie names
movies = data['title'].tolist()


from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
# import difflib

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# movies = [...]  # Your list of movies
# data = [...]  # Your movie data
# similarity_confidence = [...]  # Your similarity confidence data


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error"},
    )


@app.post("/recommendations")
async def get_movie_recommendations(request: Request):
    try:
        movie_name = await request.json()
        movie_name = movie_name.get('movie_name')

        # Find the best match for the movie name
        best_matches = difflib.get_close_matches(movie_name, movies)

        if movie_name not in best_matches:
            suggestions = [{"movie": match} for match in best_matches]
            return JSONResponse(
                content={
                    "status": "suggestions",
                    "suggested_movies": suggestions
                }
            )

        # Find the index of the movie from data
        movie_index = data[data.title == movie_name].index.values[0]

        # Get similar movies
        similar_movie_indices = list(enumerate(similarity_confidence[movie_index]))
        sorted_similar_movie_indices = sorted(similar_movie_indices, key=lambda x: x[1], reverse=True)

        recommendations = []
        for i, similar_movie_index in enumerate(sorted_similar_movie_indices[1:11], start=1):
            movie = movies[similar_movie_index[0]]
            keywords = data['keywords'][similar_movie_index[0]]
            recommendations.append({"movie": movie, "keywords": keywords})

        return JSONResponse(
            content={
                "status": "success",
                "selected_movie": movie_name,
                "recommendations": recommendations
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommendations")
async def show_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    import uvicorn
    uvicorn.run(app, host="localhost", port=5000)
