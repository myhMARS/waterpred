<template>
  <div class="relative" ref="dropdownRef">
    <button
        class="flex items-center text-gray-700 dark:text-gray-400"
        @click.prevent="toggleDropdown"
    >

      <span class="block mr-1 font-medium text-theme-sm">{{ username }}</span>

      <ChevronDownIcon :class="{ 'rotate-180': dropdownOpen }"/>
    </button>

    <!-- Dropdown Start -->
    <div
        v-if="dropdownOpen"
        class="absolute right-0 mt-[17px] flex w-[260px] flex-col rounded-2xl border border-gray-200 bg-white p-3 shadow-theme-lg dark:border-gray-800 dark:bg-gray-dark"
    >
      <div>
        <span class="block font-medium text-gray-700 text-theme-sm dark:text-gray-400">
          {{ username }}
        </span>
        <span class="mt-0.5 block text-theme-xs text-gray-500 dark:text-gray-400">
          {{ email }}
        </span>
      </div>

      <ul class="flex flex-col gap-1 pt-4 pb-3 border-b border-gray-200 dark:border-gray-800">
        <li v-for="item in menuItems" :key="item.href">
          <router-link
              :to="item.href"
              class="flex items-center gap-3 px-3 py-2 font-medium text-gray-700 rounded-lg group text-theme-sm hover:bg-gray-100 hover:text-gray-700 dark:text-gray-400 dark:hover:bg-white/5 dark:hover:text-gray-300"
          >
            <!-- SVG icon would go here -->
            <component
                :is="item.icon"
                class="text-gray-500 group-hover:text-gray-700 dark:group-hover:text-gray-300"
            />
            {{ item.text }}
          </router-link>
        </li>
      </ul>
      <router-link
          to="/"
          @click="signOut"
          class="flex items-center gap-3 px-3 py-2 mt-3 font-medium text-gray-700 rounded-lg group text-theme-sm hover:bg-gray-100 hover:text-gray-700 dark:text-gray-400 dark:hover:bg-white/5 dark:hover:text-gray-300"
      >
        <LogoutIcon
            class="text-gray-500 group-hover:text-gray-700 dark:group-hover:text-gray-300"
        />
        登出
      </router-link>
    </div>
    <!-- Dropdown End -->
  </div>
</template>

<script setup>
import {ChevronDownIcon, HomeIcon, LogoutIcon, UserCircleIcon} from '@/icons'
import {RouterLink, useRouter} from 'vue-router'
import {onMounted, onUnmounted, ref} from 'vue'
import {useAuthStore} from '@/stores/authStatus.js'
import axios from "axios";
import {showMessage} from "@/utils/vuemessage.js";

const router = useRouter()
const authStore = useAuthStore()


const dropdownOpen = ref(false)
const dropdownRef = ref(null)
const username = ref('')
const email = ref('')
const menuItems = [
  {href: '/', icon: HomeIcon, text: '主页'},
  {href: '/profile', icon: UserCircleIcon, text: '个人资料'},
]

const toggleDropdown = () => {
  dropdownOpen.value = !dropdownOpen.value
}

const closeDropdown = () => {
  dropdownOpen.value = false
}

const signOut = () => {
  authStore.setLoginStatus(false, router)
  closeDropdown()
}

const handleClickOutside = (event) => {
  if (dropdownRef.value && !dropdownRef.value.contains(event.target)) {
    closeDropdown()
  }
}

async function fetchUserData() {
  try {
    const response = await axios.get("/accounts/userinfo/", {
      headers: {
        Authorization: `JWT ${localStorage.getItem('token')}`
      }
    })
    username.value = response.data.username
    email.value = response.data.email
    localStorage.setItem('username', response.data.username)
    localStorage.setItem('email', response.data.email)
    localStorage.setItem('phone', response.data.phone)
    localStorage.setItem('userid', response.data.userid)
  } catch (error) {
    console.log(error)
  }
}

async function initAuth() {
  username.value = localStorage.getItem('username') || ''
  const currentTime = Date.now()
  const expiredTime = localStorage.getItem('expiredTime')
  const refreshToken = localStorage.getItem('refreshToken')

  if (expiredTime && Number(expiredTime) > currentTime) {
    authStore.setLoginStatus(true)
  } else if (refreshToken) {
    try {
      const response = await axios.post('/auth/jwt/refresh/', {refresh: refreshToken})
      const token = response.data.access
      localStorage.setItem('token', token)
      localStorage.setItem('expiredTime', Date.now() + 15 * 60 * 1000)
      authStore.setLoginStatus(true)
    } catch (error) {
      console.error(error)
      showMessage({
        type: 'error',
        message: '登录信息失效',
        onClose: authStore.setLoginStatus(false, router)
      })
    }
  }
}

onMounted(() => {
  initAuth()
  fetchUserData()
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>
