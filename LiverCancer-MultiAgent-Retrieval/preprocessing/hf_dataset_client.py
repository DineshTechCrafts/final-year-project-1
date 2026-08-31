"""
Client for the Hugging Face Dataset Server API.
Supports retrieving dataset information, configuration, splits, and specific rows.
Implements exponential backoff and caching to avoid repeated remote access.
"""
import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import logging
logger = logging.getLogger(__name__)

class HFDatasetClient:
    def __init__(self, dataset_name: str, base_url: str = "https://datasets-server.huggingface.co", cache_dir: str = "data/cache"):
        self.dataset_name = dataset_name
        self.base_url = base_url.rstrip('/')
        self.cache_dir = Path(cache_dir)
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup session with retry logic
        self.session = requests.Session()
        # Retry on 429 (Rate Limit), 500, 502, 503, 504
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
    def _get(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a GET request with proper error handling."""
        if params is None:
            params = {}
        params['dataset'] = self.dataset_name
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {url} with params {params}: {e}")
            raise RuntimeError(f"API request failed: {e}") from e

    def get_splits(self) -> List[Dict[str, Any]]:
        """Retrieve available splits for the dataset."""
        data = self._get("splits")
        return data.get("splits", [])
        
    def get_size(self, config: str = "default") -> Dict[str, Any]:
        """Retrieve size statistics for a specific config."""
        return self._get("size", {"config": config})
        
    def get_info(self, config: str = "default") -> Dict[str, Any]:
        """Retrieve dataset info (schema, features) for a specific config."""
        return self._get("info", {"config": config})
        
    def get_rows(self, config: str, split: str, offset: int, length: int) -> Dict[str, Any]:
        """Retrieve a specific range of rows."""
        params = {
            "config": config,
            "split": split,
            "offset": offset,
            "length": length
        }
        return self._get("rows", params)

    def get_cached_rows(self, config: str, split: str, offset: int, length: int, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Retrieve rows, using local cache to avoid duplicate downloads.
        Cache key is based on dataset, config, split, offset, and length.
        """
        # Generate cache key
        cache_key_str = f"{self.dataset_name}_{config}_{split}_{offset}_{length}"
        cache_hash = hashlib.md5(cache_key_str.encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_hash}.json"
        
        if not force_refresh and cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                return cached_data['data']
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Cache file {cache_file} corrupted, refetching. Error: {e}")
                
        # Fetch from API
        data = self.get_rows(config, split, offset, length)
        
        # Save to cache with metadata
        cache_payload = {
            "timestamp": time.time(),
            "query": {
                "dataset": self.dataset_name,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length
            },
            "data": data
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_payload, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to write cache file {cache_file}: {e}")
            
        return data
        
    def invalidate_cache(self):
        """Clear all cached responses."""
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except OSError as e:
                logger.error(f"Failed to delete cache file {cache_file}: {e}")
