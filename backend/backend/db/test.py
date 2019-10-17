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

for i in connection.find():
    print(i)

# insert

# userInfo = {
#     'name': 'jung',
#     'age': 45,
#     'tel': '015-6346-1235'
# }
#
# connection.insert(userInfo)
# docs = connection.find()

# select
print()
for user in connection.find({'name': 'choi'}):
    print(user)

# updateOne - 매칭되는 한개의 document 업데이트
result = connection.update_one(
    {'name':'jang'},   #수정할 데이터 찾을 조건
    {
        '$set':{'age': 30},
                #수정값
    })
print(result.matched_count)  #수정할 데이터 찾은 건수
print(result.modified_count)  #수정된 데이터 건수

for i in connection.find():
    print(i)
print()

# updateMany - 매칭되는 list of document 업데이트
connection.update_many({'name': 'choi'},
                      {'$set': {'age': 20}
                       })

for i in connection.find():
    print(i)

# delete
result = connection.delete_many({'age': 20})
print(result.deleted_count)

#접속 해제
client.close()