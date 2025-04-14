<template>
  <ThemeProvider>
    <SidebarProvider>
      <RouterView />
    </SidebarProvider>
  </ThemeProvider>
</template>

<script>
import ThemeProvider from './components/layout/ThemeProvider.vue'
import SidebarProvider from './components/layout/SidebarProvider.vue'
import { RouterView } from 'vue-router'
import { useAuthStore} from "@/stores/authStatus.js";



export default {
  name: 'App',
  components:{
    RouterView,
    ThemeProvider,
    SidebarProvider
  },
  beforeCreate() {
    const authStore = useAuthStore()
    authStore.initializeStore()
  }
}
</script>