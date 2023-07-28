from gmm import GMMSimple, train_model, extract_gmm_parameters
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
from fastapi.exceptions import HTTPException

app = FastAPI()
# Set up the static files (JS, CSS, images, etc.) directory
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    with open("./index.html") as f:
        html = f.read()

    return HTMLResponse(content=html, status_code=200)

#########################################################################
# ADDED BY VASU
#########################################################################

"""
Step required to be done:
1. Host the model, train and extract model parameters
2. After reading html file in line 18, parse the input points
3. Use the input points to train the model -> call the 'train_model()'
4. when type 'p', exxtract the learnt gmm parameters using 'extract_gmm_parameters()'
5. Display the model parameters using visualization
"""

# train endpoint
"""
// access endpoint /train at localhost:8000, sending the data
            response = fetch(query_string, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ coordinates: e })
"""


class coordinates(BaseModel):
    coordinates: list[dict] = None


@app.post('/train')
async def train(n_components: int, data: coordinates, epochs: int = 1000):
    body = data.json()
    model = GMMSimple(n_components=n_components)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
    loss = train_model(model=model,
                       optimizer=optimizer,
                       x=data,
                       n_epochs=epochs)
    
    # # Extract GMM parameters after training
    # data_mean = coordinates.mean()
    # data_std = coordinates.std()
    # pi, mu, sigma = extract_gmm_parameters(gmm_model, data_mean, data_std)

    # return {
    #     "loss": loss,
    #     "n_components": n_components,
    #     "pi": pi.tolist(),
    #     "mu": mu.tolist(),
    #     "sigma": sigma.tolist(),
    # }
    return {'message': 'Post successful'}





# # Sample data model for training
# class TrainingData(BaseModel):
#     data: torch.Tensor



# # Endpoint to receive data for training
# @app.post("/train")
# def train_gmm(data: TrainingData):
#     x = data.data

#     # Train the GMM model
    

    

# # load the trained model using joblib
# model = joblib.load('path_to_trained_ml_model.pkl')  # NEED to change this


# # Define the request data model using Pydantic
# class InputData(BaseModel):
#     data: list

# # Train the model
# optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# def train(data: InputData):
#     losess = train_model(model = model,
#                         optimizer = optim,
#                         x = torch.tensor(data),
#                         n_epochs = 200)


# # Save the trained model parameters


# # Define the prediction endpoint
# @app.post("/predict/")
# def predict(data: InputData):
#     try:
        
#         # Get the input data from the request
#         input_data = data.data

#         # Process the input data (if needed) and make predictions using the loaded model
#         pi, mu, sigma = model.extract_gmm_parameters(model = model, data_mean = input_data.mean(), data_std = input_data.std())

#         # Return the predictions as a dictionary
#         return {"pi": pi, 
#                 'mu': mu, 
#                 'sigma': sigma}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# # Run the application using uvicorn server
# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)
