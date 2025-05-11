import { createRouter, createWebHistory } from 'vue-router'
import HomeView from "@/views/Intruduce/HomeView.vue";
import TeamView from "@/views/Intruduce/TeamView.vue";
import DashboardView from "@/views/Manage/DashboardView.vue";
import SigninView from "@/views/Auth/SigninView.vue"
import RegisterView from "@/views/Auth/RegisterView.vue";
import {useAuthStore} from "@/stores/authStatus.js";
import UserProfileView from "@/views/User/UserProfile.vue";
import ActivateView from "@/views/Auth/ActivateView.vue";
import StationListView from "@/views/Manage/StationListView.vue";

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
      path: '/stationlist',
      name: 'stationlist',
      component: StationListView,
      meta: {
        requireLogin: true
      }
    },
    {
      path: '/login',
      name: 'login',
      component: SigninView,
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
    },
    {
      path:'/activate/:uid/:token',
      name: 'activate',
      component: ActivateView
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
