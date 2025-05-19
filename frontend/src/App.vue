<template>
  <ThemeProvider>
    <SidebarProvider>
      <RouterView />
    </SidebarProvider>
  </ThemeProvider>
</template>

<script setup>
import ThemeProvider from '@/components/layout/ThemeProvider.vue'
import SidebarProvider from '@/components/layout/SidebarProvider.vue'
import { RouterView } from 'vue-router'
import { useAuthStore } from '@/stores/authStatus.js'

// 在 setup 中使用生命周期函数初始化 store
const authStore = useAuthStore()
authStore.initializeStore()
</script>