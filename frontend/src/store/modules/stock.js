// store.js
import Vue from 'vue'
import Vuex from 'vuex'
import stockService from "../services/StockService"

Vue.use(Vuex);

export const stockStore = new Vuex.Store({
	state: {
		businessType: "",
		businessTypes: [],
		corp: "",
		corparations: [],
	},
	getters: {
		getBusinessType: function(state) {
			return state.businessType
		},
		getBusinessTypes: function(state) {
			return state.businessTypes
		}
	},
	mutations: {
		setBusinessType: function(state, payload) {
			return state.businessType = payload
		},
		setBusinessTypes: function(state, payload) {
			return state.businessTypes = payload
		}
	},
	actions: {
		loadBusinessTypes: function({commit}) {
			commit('setBusinessTypes,', stockService.loadBusinessTypes())
		}
	},
});