import Vue from 'vue'
import App from './App.vue'
import vuetify from './plugins/vuetify';
import Vuex from 'vuex'
import VueRouter from 'vue-router'
import axios from 'axios'
import VueGoogleCharts from 'vue-google-charts'

Vue.config.productionTip = false;

Vue.use(Vuex);
Vue.use(VueRouter);
Vue.use(VueGoogleCharts);
Vue.prototype.$http = axios;

new Vue({
  vuetify,
  // delimiters: ['[[', ']]'],
  render: h => h(App)
}).$mount('#app');
