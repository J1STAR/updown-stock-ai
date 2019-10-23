// stock.js
import Vue from 'vue'
import Vuex from 'vuex'
import stockService from "../../services/StockService"

Vue.use(Vuex);

const state = {
	businessTypes: [],
	corparations: [],
}

// getters
const getters = {
	getBusinessType: function(state) {
		return state.businessType
	},
	getBusinessTypes: function(state) {
		return state.businessTypes
	},
	getCorparations: function(state) {
		return state.corparations
	}
}

// actions
const actions = {
	loadBusinessTypes: async function({commit}) {
		var res = await stockService.loadBusinessTypes()
		commit('setBusinessTypes', res)
	},
	loadCorparations: async function({commit}, businessCode) {
		var res = await stockService.loadCorparations(businessCode)
		commit('setCorparations', res)
	}
}

// mutations
const mutations = {
	setBusinessType: function(state, payload) {
		return state.businessType = payload
	},
	setBusinessTypes: function(state, payload) {
		return state.businessTypes = payload
	},
	setCorparations: function(state, payload) {
		return state.corparations = payload
	}
}

export default {
	namespaced: true,
	state,
	getters,
	actions,
	mutations
}