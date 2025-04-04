<template>
  <RouterView />
</template>

<script>
import { RouterView } from 'vue-router'
import { useAuthStore} from "@/stores/authStatus.js";



export default {
  name: 'App',
  components:{
    RouterView
  },
  beforeCreate() {
    const authStore = useAuthStore()
    authStore.initializeStore()
  }
}
</script>