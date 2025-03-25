import hashlib
import json
from typing import Dict, Any, Optional
from database.db_manager import DatabaseManager
from config import CACHE_ENABLED, CACHE_EXPIRATION


class ResponseCache:
    def __init__(self, db_manager: DatabaseManager, enabled: bool = CACHE_ENABLED, 
                 expiration: int = CACHE_EXPIRATION):
        """
        Initialize the response cache.
        """
        self.db_manager = db_manager
        self.enabled = enabled
        self.expiration = expiration
    
    def get_query_hash(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a hash for a query and its parameters.
        """
        # Combine query and parameters
        if params:
            hash_input = query + json.dumps(params, sort_keys=True)
        else:
            hash_input = query
        
        # Generate hash
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def get_cached_response(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Get a cached response for a query.
        """
        if not self.enabled:
            return None
        
        query_hash = self.get_query_hash(query, params)
        return self.db_manager.get_cache_entry(query_hash)
    
    def cache_response(self, query: str, response: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Cache a response for a query.
        """
        if not self.enabled:
            return
        
        query_hash = self.get_query_hash(query, params)
        self.db_manager.add_cache_entry(query_hash, response, self.expiration)
    
    def clear_expired_cache(self) -> int:
        """
        Clear expired cache entries.
        """
        return self.db_manager.clean_expired_cache()