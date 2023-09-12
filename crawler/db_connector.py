from types import TracebackType
from typing import Optional, Type

import pymongo


class MongoContextManager:
    def __init__(self, uri: str, db_name: str, collection_name: str) -> None:
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name

    def __enter__(self) -> pymongo.collection.Collection:
        self.client: pymongo.MongoClient = pymongo.MongoClient(self.uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        return self.collection

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        self.client.close()
