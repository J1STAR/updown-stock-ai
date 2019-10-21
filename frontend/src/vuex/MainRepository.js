import MongoDBService from '../services/MongoDBService'
import store from './store'
// import router from '../routers/router'

export default {
    // 회원
    setUsers: async function () {
        store.dispatch('setUsers', await MongoDBService.getUsers());
    },
    getUsers: function () {
        return store.getters.getUsers;
    },
    postUser: async function(name, age) {
        await MongoDBService.postUser(name, age);
    }
}