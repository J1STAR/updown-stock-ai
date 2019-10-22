import MongoDBService from '../services/MongoDBService'
import store from './store'
// import router from '../routers/router'

export default {
    // 회원
    setStocks: async function () {
        store.dispatch('setStocks', await MongoDBService.getStocks());
    },
    getStocks: function () {
        return store.getters.getStocks;
    },
    postStock: async function(name, age) {
        await MongoDBService.postStock(name, age);
    },
    deleteStock: async function(stock){
        await MongoDBService.deleteStock(stock);
    }
}