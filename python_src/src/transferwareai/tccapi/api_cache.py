from typing import Optional, Any

import requests
from pathlib import Path
import polars as pl
import json
import os
import logging

from polars import DataFrame


class ApiCache:
    """Class to manage API cache"""

    @staticmethod
    def api_url(num: int | None = 1) -> str:
        """Get the TCC API URL for a given page number"""
        return (
            f"https://db.transferwarecollectorsclub.org/api/v1/patterns/page{num}?key=umich2024"
            if num > 1
            else "https://db.transferwarecollectorsclub.org/api/v1/patterns/?key=umich2024"
        )

    @staticmethod
    def get_api_page(page: int) -> Optional[list[dict[str, Any]]]:
        """
        Retrieve data from a single API page.
        :param page: page number to get.
        :return: list of json patterns.
        """
        response = requests.get(ApiCache.api_url(page))
        if response.status_code != 200:
            return None
        return response.json()

    @staticmethod
    def get_api_pages(limit: None | int = None) -> list[dict[str, Any]]:
        """
        Get all patterns from API.
        :param limit: Limit the number of pages to retrieve.
        :return list of json pattern objects.
        """
        patterns = []
        page_num = 1
        # Ends once get_api_page returns [] or error getting page
        while page := ApiCache.get_api_page(page_num):
            logging.debug(f"Retrieved page {page_num}")

            # Last page
            if len(page) == 0:
                break

            patterns.extend(page)
            page_num += 1

            if limit and page_num > limit:
                break
        return patterns

    def __init__(self, directory: Path):
        self._directory = directory
        self._cache_file = directory.joinpath(
            "cache.json"
        )  # cache file (JSON for now, upgrade to parquet later)
        self._assets_dir = directory.joinpath("assets")  # directory for images
        self._df = None

    def _requires_update(self) -> bool:
        if self._cache_file.exists():
            df = pl.read_json(self._cache_file)
            max_id_cache = df["id"].max()
            max_id_now = self.get_api_page(1)[0]["id"]

            return max_id_cache < max_id_now
        else:
            return False

    def build_cache(self):
        """Build the cache"""

        if self._requires_update():
            logging.info("Cache out of date, updating cache")

            # Get patterns JSON
            patterns = ApiCache.get_api_pages(10)

            if not self._directory.exists():
                os.makedirs(self._directory)

            # Write cache to disk
            with open(self._cache_file, "w") as buffer:
                buffer.write(json.dumps(patterns, indent=2))

        self._df = pl.read_json(self._cache_file)

        logging.info(f"Loaded cache with {len(self._df)} patterns")

        # Begin collecting assets
        if not self._assets_dir.exists():
            os.makedirs(self._assets_dir)  # create assets directory

        # Query for pattern ids + image URLs and tags
        urls = self._df.select(
            pl.col("id"),
            pl.col("images").list.eval(pl.element().struct.field("url")),
            pl.col("images").list.eval(pl.element().struct.field("tags")).alias("tags"),
        )

        # Download images
        for row in urls.iter_rows():
            pattern_id, image_urls, tags = row

            # Create patterns directory if new
            pattern_dir = self._assets_dir.joinpath(str(pattern_id))
            if not pattern_dir.exists():
                os.makedirs(pattern_dir)

            # Download all images for pattern
            for image_url, tag in zip(image_urls, tags):
                logging.debug(f"Downloading {image_url}")

                if not pattern_dir.joinpath(f"{pattern_id}-{tag}.jpg").exists():
                    image_response = requests.get(image_url)
                    image_file = pattern_dir.joinpath(f"{pattern_id}-{tag}.jpg")
                    with open(image_file, "wb") as f:
                        f.write(image_response.content)

    def as_df(self) -> DataFrame:
        return self._df
