import BootstrapVue from "bootstrap-vue"
import Vue from 'vue'
import App from './App.vue'
import router from './router'
import SmartTable from 'vuejs-smart-table'
import SortedTablePlugin from "vue-sorted-table";
import 'bootstrap/dist/css/bootstrap.css'
import 'bootstrap-vue/dist/bootstrap-vue.css'
import VueGoodTablePlugin from 'vue-good-table';
import 'vue-good-table/dist/vue-good-table.css';

Vue.use(VueGoodTablePlugin);
Vue.use(BootstrapVue)
Vue.use(SmartTable)
Vue.use(SortedTablePlugin);
Vue.config.productionTip = false

new Vue({
  router,
  render: h => h(App),
}).$mount('#app')