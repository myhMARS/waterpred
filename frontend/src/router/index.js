import {createRouter, createWebHistory} from 'vue-router'
import HomeView from "@/views/Intruduce/HomeView.vue";
import TeamView from "@/views/Intruduce/TeamView.vue";
import DashboardView from "@/views/Manage/DashboardView.vue";
import SigninView from "@/views/Auth/SigninView.vue"
import RegisterView from "@/views/Auth/RegisterView.vue";
import {useAuthStore} from "@/stores/authStatus.js";
import UserProfileView from "@/views/User/UserProfile.vue";
import ActivateView from "@/views/Auth/ActivateView.vue";
import StationListView from "@/views/Manage/StationListView.vue";
import StationdetailView from "@/views/Manage/StationdetailView.vue";
import StationMap from "@/views/Manage/StationMap.vue";
import AreaMap from "@/views/Manage/AreaMap.vue";
import AreaListView from "@/views/Manage/AreaListView.vue";
import AreadetialView from "@/views/Manage/AreadetialView.vue";
import NotFound from "@/views/FourZeroFour.vue"

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
            path: '/activate/:uid/:token',
            name: 'activate',
            component: ActivateView
        },
        {
            path: '/station/:id',
            name: 'stationdetail',
            component: StationdetailView
        },
        {
            path: '/stationmap',
            name: 'stationmap',
            component: StationMap
        },
        {
            path: '/areamap',
            name: 'areamap',
            component: AreaMap
        },
        {
            path: '/arealist',
            name: '/arealist',
            component: AreaListView
        },
        {
            path: '/area/:county',
            name: 'areadetail',
            component: AreadetialView
        },
        {
            path: '/:pathMatch(.*)*',
            name: 'notfound',
            component: NotFound
        }
    ],
})

router.beforeEach((to, from, next) => {
    const authStore = useAuthStore()
    authStore.initializeStore()
    if (authStore.isLogin && (to.name === 'login' || to.name === 'register')) {
        next({name: 'home'})
    } else if (to.matched.some(record => record.meta.requireLogin) && !authStore.isLogin) {
        next({name: 'login', query: {jump: to.path}})
    } else {
        next()
    }
})

export default router
