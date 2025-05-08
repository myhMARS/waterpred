import { createRouter, createWebHistory } from 'vue-router'
import HomeView from "@/views/HomeView.vue";
import TeamView from "@/views/TeamView.vue";
import DashboardView from "@/views/DashboardView.vue";
import Signin from "@/views/Auth/Signin.vue"
import RegisterView from "@/views/RegisterView.vue";
import {useAuthStore} from "@/stores/authStatus.js";
import UserProfileView from "@/views/User/UserProfile.vue";

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
      path: '/dashboard',
      name: 'dashboard',
      component: DashboardView,
      meta: {
        requireLogin: true
      }
    },
    {
      path: '/login',
      name: 'login',
      component: Signin,
    },
    {
      path: '/register',
      name: 'register',
      component: RegisterView
    },
    {
      path: '/profile',
      name: 'profile',
      component: UserProfileView,
      meta: {
        requireLogin: true
      }
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
