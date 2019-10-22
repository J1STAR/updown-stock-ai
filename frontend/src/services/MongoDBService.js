import axios from 'axios'
import moment from 'moment'

export default {
    getStocks: function () {
        return axios.get('api/stocks/')
            .then(response => response.data.data)
            .catch(error => console.log(error));
    },
    postStock: function (stock) {
        console.log("stock: ", stock);
        console.log("날짜: ", moment(new Date()).format('YYYY-MM-DD'));
        return axios
            .post(
                'api/stocks/',
            {
                date: moment(new Date()).format('YYYY-MM-DD'),
                start_price: stock.start_price,
                end_price: stock.end_price,
                high_price: stock.high_price,
                low_price: stock.low_price,
                val: stock.val
            })
            .then(response => response.data)
            .catch(error => console.log(error));
    },
    deleteStock: function (stock) {
        const url = `api/stocks/${stock.pk}`;
        return axios.delete(url)
            .then(response => response.data)
            .catch(error => console.log(error));
    }
}