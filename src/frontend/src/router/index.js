import Vue from 'vue'
import Router from 'vue-router'
import XAI from '../components/Xai.vue'
import Exploit from '../components/Exploit.vue'

import Home from '../components/Home.vue'
import FAQs from '../components/FAQs.vue';
const DEFAULT_TITLE = 'xaiploit';

Vue.use(Router)

const router = new Router({
  mode: "history",
  base: process.env.BASE_URL,
  routes: [
    {
      path: "/",
      name: "Home",
      component: Home,
    },
    {
      path: "/predictions",
      name: "ExploitPrediction",
      component: Exploit,
    },
    {
      path: '/explain',
      name: 'Explain',
      component: XAI,
    },
    {
      path: '/faqs',
      name: 'FAQs',
      component: FAQs,
    },
  ],
});
router.afterEach(() => {
  // Use next tick to handle router history correctly
  // see: https://github.com/vuejs/vue-router/issues/914#issuecomment-384477609
  Vue.nextTick(() => {
    document.title = DEFAULT_TITLE;
  });
});
export default router;
