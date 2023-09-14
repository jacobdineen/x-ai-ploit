import Vue from 'vue'
import Router from 'vue-router'
import Home from '../components/Home.vue'
const DEFAULT_TITLE = 'OSS Data Explorer';

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