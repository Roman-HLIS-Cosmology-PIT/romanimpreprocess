import os
import requests
import roman_datamodels as rdm
from astroquery.mast import MastMissions
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from dotenv import load_dotenv


if __name__ == "__main__":  # pragma: no cover
    _ = load_dotenv(dotenv_path="./variables.env")

    # Create MastMissions object and assign mission to 'roman'
    missions = MastMissions(mission='roman')

    # Login to search and retrieve Roman data
    token = os.getenv("MAST_API_TOKEN")    
    missions.login(token=token)
