from mongoengine import Document, fields

# Create your models here.


class Stock(Document):
    date = fields.DateField(required=True)
    start_price = fields.IntField(required=True)
    end_price = fields.IntField(required=True)
    high_price = fields.IntField(required=True)
    low_price = fields.IntField(required=True)
    val = fields.IntField(required=True)
