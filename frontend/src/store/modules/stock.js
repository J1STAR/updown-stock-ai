// stock.js
import Vue from 'vue'
import Vuex from 'vuex'
import stockService from "../../services/StockService"

Vue.use(Vuex);

const state = {
	businessTypes: [],
	corporations: [],
}

// getters
const getters = {
	getBusinessType: function(state) {
		return state.businessType
	},
	getBusinessTypes: function(state) {
		return state.businessTypes
	},
	getCorporations: function(state) {
		return state.corporations
	}
}

// actions
const actions = {
	loadBusinessTypes: async function({commit}) {
		let res = await stockService.loadBusinessTypes()
		commit('setBusinessTypes', res)
	},
	loadCorporations: async function({commit}, businessCode) {
		let res = await stockService.loadCorporations(businessCode)
		commit('setCorporations', res)
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
	setCorporations: function(state, payload) {
		return state.corporations = payload
	}
}

export default {
	namespaced: true,
	state,
	getters,
	actions,
	mutations
}