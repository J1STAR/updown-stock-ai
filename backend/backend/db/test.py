#파이썬으로 MongoDB에 데이터 저장하기

from pymongo import MongoClient   #mongodb 모듈 지정
import datetime
import pprint

from bson.objectid import ObjectId  #objectid 모듈 지정

#mongodb 연결객체 생성
# client = MongoClient()
# client = MongoClient('192.168.19.132', '27017')  #접속IP, 포트
client = MongoClient('mongodb://localhost:27017/')
print(client)
db = client.sample

connection = db.user
docs = connection.find()

for i in docs:
    print(i)

#접속 해제
client.close()