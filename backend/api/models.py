from mongoengine import Document, fields


class User(Document):
    name = fields.StringField(required=True)
    age = fields.IntField(required=True)


class InputStock(Document):
    date = fields.DateField(required=True)
    startprice = fields.IntField(required=True)
    endprice = fields.IntField(required=True)
    highprice = fields.IntField(required=True)
    lowprice = fields.IntField(required=True)
    val = fields.IntField(required=True)


class OutputStock(Document):
    date = fields.DateField(required=True)
    price = fields.IntField(required=True)
    predictdate = fields.DateField(required=True)
    predictprice = fields.IntField(required=True)
