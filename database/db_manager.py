from pymongo import MongoClient
from urllib.parse import quote_plus
from datetime import datetime, timedelta
from config import DATABASE_NAME, DB_USERNAME, DB_PASSWORD, DB_HOST, DB_PORT

class DatabaseManager:
    def __init__(self):
        """Initialize the database manager with MongoDB connection details."""
        encoded_username = quote_plus(DB_USERNAME)
        encoded_password = quote_plus(DB_PASSWORD)  # Encodes special characters properly

        mongo_uri = f"mongodb://{encoded_username}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DATABASE_NAME}?authSource=admin"

        self.client = MongoClient(mongo_uri)
        self.db = self.client[DATABASE_NAME]

    def add_cache_entry(self, query_hash, response, expiration_seconds=3600):
        """Add a cache entry for a query response."""
        expiration_time = datetime.now() + timedelta(seconds=expiration_seconds)
        self.db.response_cache.update_one(
            {"query_hash": query_hash},
            {"$set": {"response": response, "timestamp": datetime.now(), "expiration": expiration_time}},
            upsert=True
        )

    def get_cache_entry(self, query_hash):
        """Get a cached response for a query."""
        result = self.db.response_cache.find_one({"query_hash": query_hash, "expiration": {"$gte" : datetime.now()}})
        if result:
            return result["response"]
        return None

    def clean_expired_cache(self):
        """Remove expired cache entries."""
        result = self.db.response_cache.delete_many({"expiration": {"$lt": datetime.now()}})
        return result.deleted_count
    
    def clean_all_cache(self):
        """Remove all cache entries."""
        if self.db.response_cache is not None:
            result = self.db.response_cache.delete_many({})
            return result.deleted_count
        else:
            return 0
        
    def close_connection(self):
        """Close the database connection."""
        self.client.close()
