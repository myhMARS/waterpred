import { defineStore } from 'pinia'

export const useAuthStore = defineStore('auth', {
  state: () => ({
    isLogin: false
  }),
  actions: {
    initializeStore() {
      this.isLogin = !!localStorage.getItem('token');
    },
    setLoginStatus(status) {
      this.isLogin = status
      if (!status) {
        localStorage.clear()
      }
    }
  }
})
