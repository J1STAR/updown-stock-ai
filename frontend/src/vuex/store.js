import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex);

export default new Vuex.Store({
    state: {
        stocks: [],
    },

    getters: {
        getStocks: state => {
            return state.stocks;
        }
    },

    mutations: {
        setStocks(state, payload) {
            console.log(payload);
            state.stocks = payload;
        }
    },

    actions: {
        setStocks({commit}, response) {
            commit('setStocks', response);
        }
    }
})