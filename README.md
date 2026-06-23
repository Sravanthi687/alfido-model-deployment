# Alfido Tech Model Deployment

This repository contains the solution for the "Model Deployment - API Containerization" task. It wraps a trained Machine Learning model (Logistic Regression on the Iris dataset) inside a **FastAPI** application and containerizes it using **Docker**.

## Project Files
- `train_model.py`: Script that trains the scikit-learn model and saves it as `model.pkl`.
- `app.py`: The FastAPI server that loads the model and exposes a `/predict` endpoint.
- `requirements.txt`: Python package dependencies.
- `Dockerfile`: Configuration for building the Docker image.

---

## 1. Running Locally with Docker

To build and run this API locally, make sure you have [Docker](https://www.docker.com/) installed and running on your system.

### Build the Docker Image
Open your terminal in this directory (`alfido-model-deployment`) and run:
```bash
docker build -t alfido-model-api .
```
*(Note: The `Dockerfile` automatically runs `train_model.py` during the build process to generate the `model.pkl` file directly inside the container!)*

### Run the Docker Container
Once built, start the container and map port 8000:
```bash
docker run -d -p 8000:8000 --name alfido-api alfido-model-api
```

---

## 2. API Documentation & Testing

FastAPI automatically generates an interactive UI. Once the container is running, you can visit:
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Health Check**: [http://localhost:8000/](http://localhost:8000/)

### Example Request (Using cURL)

You can test the inference endpoint from your command line using `curl`:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

### Expected Response

The model evaluates the features and returns the predicted class index and name (e.g., `setosa`).

```json
{
  "prediction_class_index": 0,
  "prediction_class_name": "setosa"
}
```

---

## 3. Stopping the Container

To stop and remove the container when you are finished:
```bash
docker stop alfido-api
docker rm alfido-api
```
#PROJECT DEMO VIDEO:
https://www.linkedin.com/posts/sravanthi-v-880002415_internspark-aiinternship-machinelearning-ugcPost-7475021337724506112-V79X/?utm_source=share&utm_medium=member_desktop&rcm=ACoAAGmKomAB9pHnMjdMezEoRyLTMuniH02ChZE
