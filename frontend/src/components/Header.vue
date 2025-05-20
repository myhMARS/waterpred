<template>
  <header class="fixed inset-x-0 top-0 z-50">
    <nav class="mx-auto flex max-w-7xl items-center justify-between p-6 lg:px-8" aria-label="Global">
      <div class="flex lg:flex-1">
        <a href="#" class="-m-1.5 p-1.5">
          <span class="sr-only">Our Team</span>
          <img class="h-8 w-auto" src="@/assets/logo.svg" alt="" />
        </a>
      </div>
      <div class=" flex lg:hidden">
        <button type="button" class="-m-2.5 inline-flex items-center justify-center rounded-md p-2.5 text-gray-700"
          @click="mobileMenuOpen = true">
          <span class="sr-only">Open main menu</span>
          <Bars3Icon class="size-6" aria-hidden="true" />
        </button>
      </div>
      <div class="hidden lg:flex lg:gap-x-12">
        <a v-for="item in navigation" :key="item.name" :href="item.href"
          class="text-sm/6 font-semibold text-gray-900">{{ item.name }}</a>
      </div>
      <Popover v-if="authStore.isLogin" class="hidden lg:flex lg:flex-1 lg:justify-end">
        <!-- 父容器添加 relative 确保定位基准 -->
        <div class="relative">
          <!-- 显示用户名，点击打开下拉菜单 -->
          <PopoverButton class="text-left inline-flex focus:outline-none">
            <span class="text-sm font-semibold text-gray-900 block">
              {{ username }}
            </span>
            <ChevronDownIcon class="size-5" aria-hidden="true" />

          </PopoverButton>

          <!-- 下拉菜单（PopoverPanel） -->
          <transition enter-active-class="transition ease-out duration-200" enter-from-class="opacity-0 translate-y-1"
            enter-to-class="opacity-100 translate-y-0" leave-active-class="transition ease-in duration-150"
            leave-from-class="opacity-100 translate-y-0" leave-to-class="opacity-0 translate-y-1">
            <!-- 调整定位和宽度 -->
            <PopoverPanel class="absolute left-0 z-10 mt-2 w-30 min-w-full origin-top-left">
              <div class="w-full rounded-lg bg-white shadow-lg ring-1 ring-gray-900/5 overflow-hidden">
                <!-- 菜单项 -->
                <div class="relative p-4 hover:bg-gray-50 cursor-pointer">
                  <a @click.prevent="logout" class="font-semibold text-sm text-gray-900 block">注销登录</a>
                </div>
              </div>
            </PopoverPanel>
          </transition>
        </div>
      </Popover>
      <div v-else class="hidden lg:flex lg:flex-1 lg:justify-end">
        <a href="/login" class="text-sm/6 font-semibold text-gray-900">登录<span aria-hidden="true">&rarr;</span></a>
      </div>

    </nav>
    <Dialog class="lg:hidden" @close="mobileMenuOpen = false" :open="mobileMenuOpen">
      <div class="fixed inset-0 z-50" />
      <DialogPanel
        class="fixed inset-y-0 right-0 z-50 w-full overflow-y-auto bg-white px-6 py-6 sm:max-w-sm sm:ring-1 sm:ring-gray-900/10">
        <div class="flex items-center justify-between">
          <a href="#" class="-m-1.5 p-1.5">
            <span class="sr-only">Your Company</span>
            <img class="h-8 w-auto" src="@/assets/logo.svg" alt="" />
          </a>
          <button type="button" class="-m-2.5 rounded-md p-2.5 text-gray-700" @click="mobileMenuOpen = false">
            <span class="sr-only">Close menu</span>
            <XMarkIcon class="size-6" aria-hidden="true" />
          </button>
        </div>
        <div class="mt-6 flow-root">
          <div class="-my-6 divide-y divide-gray-500/10">
            <div class="space-y-2 py-6">
              <a v-for="item in navigation" :key="item.name" :href="item.href"
                class="-mx-3 block rounded-lg px-3 py-2 text-base/7 font-semibold text-gray-900 hover:bg-gray-50">{{
                  item.name
                }}</a>
            </div>
            <div v-if="authStore.isLogin">
              <a class="-mx-3 block rounded-lg px-3 py-2.5 text-base/7 font-semibold text-gray-900 hover:bg-gray-50">
                {{ username }}
              </a>
              <a @click.prevent="logout"
                class="-mx-3 block rounded-lg px-3 py-2.5 text-base/7 font-semibold text-gray-900 hover:bg-gray-50">
                退出登录
              </a>
            </div>
            <div v-else class="py-6">
              <a href="/login"
                class="-mx-3 block rounded-lg px-3 py-2.5 text-base/7 font-semibold text-gray-900 hover:bg-gray-50">Log
                in</a>
            </div>
          </div>
        </div>
      </DialogPanel>
    </Dialog>
  </header>
</template>

<script>
import { ref } from 'vue'
import { Dialog, DialogPanel, Popover, PopoverButton, PopoverPanel } from '@headlessui/vue'
import { Bars3Icon, XMarkIcon } from '@heroicons/vue/24/outline'
import { ChevronDownIcon } from '@heroicons/vue/20/solid'
import { useAuthStore } from "@/stores/authStatus.js"
import axios from "axios";

export default {
  name: 'Header',
  methods: {
    useAuthStore,
    logout() {
      this.authStore.setLoginStatus(false, this.$router)
    }
  },
  components: {
    PopoverPanel,
    Popover,
    PopoverButton,
    Dialog,
    DialogPanel,
    Bars3Icon,
    XMarkIcon,
    ChevronDownIcon
  },
  data() {
    return {
      navigation: [
        { name: '首页', href: '/' },
        { name: '监控', href: '/dashboard' },
        { name: '团队', href: '/team' },
      ],
      mobileMenuOpen: ref(false),
      username: ''
    }
  },
  computed: {
    // 让模板可以直接使用 authStore 的数据
    authStore() {
      return useAuthStore()
    },
  },
  created() {
    this.username = localStorage.getItem('username') || ''
    const currentTime = Date.now()
    const expiredTime = localStorage.getItem("expiredTime")
    const refreshToken = localStorage.getItem("refreshToken")
    if (expiredTime > currentTime) {
      this.authStore.setLoginStatus(true)
    }
    else if (refreshToken) {
      axios
        .post("auth/jwt/refresh/", { "refresh": refreshToken })
        .then(response => {
          const token = response.data.access
          localStorage.setItem("token", token)
          localStorage.setItem("expiredTime", Date.now() + 15 * 60 * 1000)
        })
        .catch(error => {
          console.log(error)
          this.authStore.setLoginStatus(false, this.$router)
        })
    }
  },
}
</script>
