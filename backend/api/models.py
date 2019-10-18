from mongoengine import Document, fields

# Create your models here.


class User(Document):
    name = fields.StringField(required=True)
    age = fields.IntField(required=True)
