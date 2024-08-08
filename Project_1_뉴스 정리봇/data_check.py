from pymongo import MongoClient

client = MongoClient(host="localhost", port=27017)
db = client["project1"]
collection = db["NewsAnalysis"]

for item in collection.find():
    print(item)
