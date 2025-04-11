import { createRouter, createWebHistory } from 'vue-router'
import HomeView from "@/views/HomeView.vue";
import TeamView from "@/views/TeamView.vue";
import DemoView from "@/views/DemoView.vue";
import LoginView from "@/views/LoginView.vue";
import RegisterView from "@/views/RegisterView.vue";
import {useAuthStore} from "@/stores/authStatus.js";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView,
    },
    {
      path: '/team',
      name: 'team',
      component: TeamView,
    },
    {
      path: '/demo',
      name: 'demo',
      component: DemoView,
      meta: {
        requireLogin: true
      }
    },
    {
      path: '/login',
      name: 'login',
      component: LoginView,
    },
    {
      path: '/register',
      name: 'register',
      component: RegisterView
    }
  ],
})

router.beforeEach((to, from, next) => {
  const authStore = useAuthStore()
  authStore.initializeStore()
  if (authStore.isLogin && (to.name === 'login' || to.name === 'register')){
    next({name: 'home'})
  }
  else if (to.matched.some(record => record.meta.requireLogin) && !authStore.isLogin) {
    next({name: 'login', query: {jump: to.path}})
  }
  else {
    next()
  }
})

export default router
