from typing import Optional

import requests
from io import StringIO
from pathlib import Path
import polars as pl
import json
import os



class ApiCache:
    """Class to manage API cache"""
    
    @staticmethod
    def api_url(num: int | None = 1) -> str:
        """Get the TCC API URL for a given page number"""
        return f"https://db.transferwarecollectorsclub.org/api/v1/patterns/page{num}?key=umich2024" if num > 1 \
                else "https://db.transferwarecollectorsclub.org/api/v1/patterns/?key=umich2024"
    
    @staticmethod
    def get_api_page(page: int) -> Optional[list]:
        """Retrieve data from API"""
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
        while (page := ApiCache.get_api_page(page_num)):
            if page is None: # error retrieving page
                break
            
            patterns.extend(page)
            page_num += 1
        return patterns

    def __init__(self, directory: Path):
        self._directory = directory
        self._cache_file = directory.joinpath("cache.json")  # cache file (JSON for now, upgrade to parquet later)
        self._assets_dir = directory.joinpath("assets")  # directory for images

    
    def build_cache(self):
        """Build the cache"""

        patterns = ApiCache.get_api_pages()
        # Convert to file-like object and then to polars dataframe:
        # For now, use JSON to easily monitor cache; upgrade to parquet later
        # using StringIO to avoid extra file I/O
        buffer = open(self._cache_file, "w")
        buffer.write(json.dumps(patterns, indent=2))
        df = pl.DataFrame(buffer)

        # Write dataframe to parquet file (for now, use JSON for readability)
        # df.write_parquet(self._cache_file)

        # Begin collecting assets
        if not self._assets_dir.exists():
            os.makedirs(self._assets_dir)  # create assets directory
        # Query for pattern ids + image URLs and tags
        urls = df.select(pl.col("id"), pl.col("images").list.eval(pl.element().struct.field("url")), pl.col("images").list.eval(pl.element().struct.field("tags")))
        for row in urls.iter():
            pattern_id, image_urls, tags = row
            pattern_dir = self._assets_dir.joinpath(str(pattern_id))
            if not pattern_dir.exists():
                os.makedirs(pattern_dir)
            for image_url, tag in zip(image_urls, tags):
                image_response = requests.get(image_url)
                image_file = pattern_dir.joinpath(f"{pattern_id}-{tag}.jpg")
                with open(image_file, 'wb') as f:
                    f.write(image_response.content)
