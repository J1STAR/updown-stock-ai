import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

export default new Vuex.Store({
    state: {
        users: [],
    },

    getters: {
        getUsers: state => {
            return state.users;
        }
    },

    mutations: {
        setUsers(state, payload) {
            state.users = payload;
        }
    },

    actions: {
        setUsers({commit}, response) {
            commit('setUsers', response);
        }
    }
})