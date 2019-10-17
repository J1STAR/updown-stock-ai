import Vue from 'vue'
import App from './App.vue'
import vuetify from './plugins/vuetify';
import Vuex from 'vuex'
import VueRouter from 'vue-router'
import axios from 'axios'

import router from './routers/router'
import store from './vuex/store'

Vue.config.productionTip = false;

Vue.use(Vuex);
Vue.use(VueRouter);
Vue.prototype.$http = axios;

new Vue({
  vuetify,
  router,
  store,
  // delimiters: ['[[', ']]'],
  render: h => h(App)
}).$mount('#app');
