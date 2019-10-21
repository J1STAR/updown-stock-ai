from mongoengine import EmbeddedDocument, Document, fields


class StockInfo(EmbeddedDocument):
    date = fields.DateTimeField()
    closing_price = fields.IntField()
    diff = fields.IntField()
    open_price = fields.IntField()
    high_price = fields.IntField()
    low_price = fields.IntField()
    volumn = fields.IntField()
    meta = {
        'ordering': ['-date']
    }


class Corp(EmbeddedDocument):
    corp_name = fields.StringField()
    stock_info = fields.EmbeddedDocumentListField(StockInfo)


# Create your models here.
class Stock(Document):
    code = fields.StringField()
    corp = fields.EmbeddedDocumentField(Corp)
