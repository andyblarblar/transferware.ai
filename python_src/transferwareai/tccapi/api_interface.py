from typing import Optional

import requests
import polars as pl
from pathlib import Path
import os


def get_data(url: str) -> Optional[dict]:
    """Retrieve data from API"""
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()


def get_transferware_page(name: str) -> list[dict]:
    """Get all records from a given table"""
    url = "https://db.transferwarecollectorsclub.org/api/v1/%s" % name
    objects = []  # list of json objects
    page = 1  # page number

    # Loop until we don't get results for a page
    while True:
        # get the next page of records
        response = get_data(f"{url}?page={page}")
        if not response:  # no more records
            break
        # get block of records from response
        block = response.get("records", [])  # TODO this prob isnt required
        objects.extend(block)  # add block to list of objects
        page += 1

    return objects


def get_transferware_patterns():
    return get_transferware_page("patterns")


class APICache:
    def __init__(self, directory: Path):
        self._directory = directory
        self._db_file = directory.joinpath("db.csv")  # csv cache file
        self._assets_dir = directory.joinpath("assets")  # directory for images

    def ensure_cache(self):
        patterns = get_transferware_patterns()  # get patterns from API
        df = pl.DataFrame(patterns)  # convert to polars dataframe
        df.write_csv(self._db_file)  # write dataframe to csv

        if not self._assets_dir.exists():
            os.makedirs(self._assets_dir)  # create assets directory

        # TODO: download images and store in assets directory
        # for pattern in patterns:
        #     pattern_id = pattern[<<whatever 'id' is stored as>>] # get pattern id
        #     pattern_dir = os.path.join(self.assets_dir, str(pattern_id)) # directory for pattern images
        #     if not os.path.exists(pattern_dir):
        #         os.makedirs(pattern_dir)
        #     if pattern['center'] is not None:
        #         image_file = os.path.join(pattern_dir, f'{pattern_id}-center.jpg')
        #         urlretrieve(pattern['center'], image_file)
