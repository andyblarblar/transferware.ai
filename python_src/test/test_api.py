import pytest
from transferwareai.tccapi.api_cache import ApiCache


class TestApi:
    @pytest.fixture
    def api_client(self, tmp_path):
        api = ApiCache(tmp_path)
        return api

    @pytest.mark.parametrize("num", (1, 2, 100))
    def test_page(self, api_client, num):
        assert ApiCache.get_api_page(num) is not None

    def test_end_page(self, api_client):
        assert ApiCache.get_api_page(999999999) == []
