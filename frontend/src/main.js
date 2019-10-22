import Vue from 'vue'
import App from './App.vue'
import vuetify from './plugins/vuetify';
import VueRouter from 'vue-router'
import axios from 'axios'
import VueGoogleCharts from 'vue-google-charts'

import store from "@/store";

Vue.config.productionTip = false;

Vue.use(VueRouter);
Vue.use(VueGoogleCharts);
Vue.prototype.$http = axios;

new Vue({
  vuetify,
  store,
  // delimiters: ['[[', ']]'],
  render: h => h(App)
}).$mount('#app');
