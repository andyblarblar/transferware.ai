from typing import Optional

import requests
from pathlib import Path
import polars as pl
import json
import os
import logging


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
    def get_api_page(page: int) -> Optional[list]:
        """Retrieve data from API. Returns JSON for some patterns. Empty list if out of range."""
        response = requests.get(ApiCache.api_url(page))
        if response.status_code != 200:
            return None
        return response.json()

    @staticmethod
    def get_api_pages() -> list:
        """Get all patterns from API"""
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
        return patterns

    def __init__(self, directory: Path):
        self._directory = directory
        self._cache_file = directory.joinpath(
            "cache.json"
        )  # cache file (JSON for now, upgrade to parquet later)
        self._assets_dir = directory.joinpath("assets")  # directory for images

    def build_cache(self):
        """Build the cache"""

        # TODO first find if cache needs to be updated

        patterns = ApiCache.get_api_pages()
        # Convert to file-like object and then to polars dataframe:
        # For now, use JSON to easily monitor cache; upgrade to parquet later
        # using StringIO to avoid extra file I/O
        buffer = open(self._cache_file, "w")
        buffer.write(json.dumps(patterns, indent=2))
        df = pl.DataFrame(buffer)  # TODO just read from cache file?

        # Write dataframe to parquet file (for now, use JSON for readability)
        # df.write_parquet(self._cache_file)

        # Begin collecting assets
        if not self._assets_dir.exists():
            os.makedirs(self._assets_dir)  # create assets directory

        # Query for pattern ids + image URLs and tags
        urls = df.select(
            pl.col("id"),
            pl.col("images").list.eval(pl.element().struct.field("url")),
            pl.col("images").list.eval(pl.element().struct.field("tags")),
        )  # TODO null tags?

        for row in urls.iter_rows():
            pattern_id, image_urls, tags = row

            # Create patterns directory if new
            pattern_dir = self._assets_dir.joinpath(str(pattern_id))
            if not pattern_dir.exists():
                os.makedirs(pattern_dir)

            # Download all images for pattern
            for image_url, tag in zip(image_urls, tags):
                logging.debug(f"Downloading {image_url}")
                image_response = requests.get(image_url)
                image_file = pattern_dir.joinpath(f"{pattern_id}-{tag}.jpg")
                with open(image_file, "wb") as f:
                    f.write(image_response.content)
