from gmm import GMMSimple, train_model, extract_gmm_parameters
import torch
import numpy
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from logs import setup_logger
from pathlib import Path
import json

app = FastAPI()
# Set up the static files (JS, CSS, images, etc.) directory
app.mount("/static", StaticFiles(directory="static"), name="static")
# create a "/static/logs.txt" file
#######################################
# Pending: should create logs.txt if it doesn't exist
#######################################
file_path = Path("./static/logs.txt")
with open(file_path, 'w') as f:
    logger = setup_logger(file_path)


@app.get("/")
async def root():
    try:
        with open("./index.html") as f:
            html = f.read()

        return HTMLResponse(content=html, status_code=200)
    except BaseException as e:
        logger.warning(f'Encountered exception when loading application: {e}')


class coordinates(BaseModel):
    coordinates: list[dict]


# train endpoint
@app.post('/train')
async def train(n_components: int, body: coordinates, epochs: int = 1000):
    logger.info(f'Input for training are: \n\tn_components: {n_components} \n\tepochs: {epochs} \n\t{body}')
    # extract the coordinates from 'body' and pass the 'x' tensor to gmm model
    try:
        coordinates = body.coordinates
        x_coordinates = [coordinate['x'] for coordinate in coordinates]

        # convert x_coordinates to x: torch.tensor type
        x = torch.tensor(x_coordinates, dtype=torch.float32)
        logger.info(f'Extracted x coordinates are: \n\t{x}')
    except BaseException as e:
        logger.warning(f'Encountered exception when extracting coordinates: {e}')

    # host, train GMM model using 'x' torch.tensor and extract learnt model's parameter
    try:
        model = GMMSimple(n_components=n_components)
        parameters = model.pi, model.mu, model.sigma
        optimizer = torch.optim.Adam(parameters, lr=1e-3,
                                     betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=0)
        logger.info(f'Model confingurations: \n\tmodel: {model}, \n\tpi: {model.pi}, \n\tmu: {model.mu}, \n\tsigma: {model.sigma}, \n\t{optimizer}')
        loss = train_model(model=model,
                           optimizer=optimizer,
                           x=x,
                           n_epochs=epochs)

        logger.info(f'GMM model loss: {loss}')
        # Extract GMM parameters after training
        data_mean = float(x.mean())
        data_std = float(x.std())

        pi, mu, sigma = extract_gmm_parameters(model=model,
                                               data_mean=data_mean,
                                               data_std=data_std)
        logger.info(f'GMM model parameters are: \n\tpi:{pi}, \n\tmu: {mu}, \n\tsigma: {sigma}')

        json_response = json.dumps({'pi': pi.tolist(),
                                    'mu': mu.tolist(),
                                    'sigma': sigma.tolist()})
        return JSONResponse(content=json_response)
    except BaseException as e:
        logger.warning(f'Encountered exception when retrieving model parameters: {e}')
        return HTMLResponse(content="Model Parameters Not Found", status_code=404)
