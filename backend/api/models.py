from mongoengine import Document, fields


class User(Document):
    name = fields.StringField(required=True)
    age = fields.IntField(required=True)

