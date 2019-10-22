import axios from 'axios'

export default {
	loadBusinessTypes: async function() {
		return await axios.get('/stock/businessTypes').then((res) => {
			return res.data
		})
	},
	loadCorparations: async function(businessCode) {
		return await axios.get('/stock/businessTypes/' + businessCode).then((res) => {
			return res.data
		})
	}
}