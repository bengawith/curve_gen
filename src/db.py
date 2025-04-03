from pymongo import MongoClient, errors
import os

env = os.getenv('.env')


class DataDB:
    MODEL_COLL = "top_models"
    TUNE_COLL = "tuning"

    def __init__(self):
        self.client = MongoClient(env.MONGO_STR)
        self.db = self.client.project
        #self.test_connection()

    def test_connection(self):
        print(self.db.list_collection_names())

    
    def delete_many(self, collection, **kwargs):
        try:
           _ = self.db[collection].delete_many(kwargs)
        except errors.InvalidOperation as e:
            print(f"delete_many error: {e}")
    

    def delete_one(self, collection, **kwargs):
        try:
           _ = self.db[collection].delete_one(kwargs)
        except errors.InvalidOperation as e:
            print(f"delete_one error: {e}")


    def add_one(self, collection, ob):
        try:
           _ = self.db[collection].insert_one(ob)
        except errors.InvalidOperation as e:
            print(f"add_one error: {e}")


    def add_many(self, collection, ob_list):
        try:
           _ = self.db[collection].insert_many(ob_list)
        except errors.InvalidOperation as e:
            print(f"add_many error: {e}")
        

    def query_distinct(self, collection, key):
        try:
           return self.db[collection].distinct(key)
        except errors.InvalidOperation as e:
            print(f"query_distinct error: {e}")


    def query_single(self, collection, **kwargs):
        try:
           r = self.db[collection].find_one(kwargs, {'_id': 0})
           return r
        except errors.InvalidOperation as e:
            print(f"query_single error: {e}")


    def query_all(self, collection, **kwargs):
        try:
           data = []
           r = self.db[collection].find(kwargs, {'_id': 0})
           for item in r:
               data.append(item)
           return data        
        except errors.InvalidOperation as e:
            print(f"query_all error: {e}")