var express = require('express');
var router = express.Router();
var request = require('request');

/* GET users listing. */
router.get('/', function(req, res, next) {
	request('https://finance.naver.com/item/sise_day.nhn?code=005930&page=1', function(error, response, body){
		console.log(body)
	})
	res.send('respond with a resource2');
});

module.exports = router;
