<template>
  <Header/>
  <section class="grid text-center h-screen items-center p-8">
    <div>
      <h3 class="text-3xl font-semibold text-blue-gray-900 mb-2">注 册</h3>

      <form @submit.prevent="register" class="mx-auto max-w-[24rem] text-left">
        <div class="mb-6">
          <label for="username" class="block font-medium text-gray-900 mb-2">用户名</label>
          <input
              id="username"
              type="text"
              v-model="username"
              placeholder="Enter your username"
              class="w-full h-11 px-3 py-3 rounded-md border border-blue-gray-200 focus:border-gray-900"
              autocomplete="username"
              required
          />
        </div>
        <div class="mb-6">
          <label for="password" class="block font-medium text-gray-900 mb-2">密码</label>
          <input
              id="password"
              type="password"
              v-model="password"
              placeholder="********"
              class="w-full h-11 px-3 py-3 rounded-md border border-blue-gray-200 focus:border-gray-900"
              autocomplete="new-password"
              required
          />
        </div>
        <div class="mb-6">
          <label for="password" class="block font-medium text-gray-900 mb-2">再次输入密码</label>
          <input
              id="re_password"
              type="password"
              v-model="re_password"
              placeholder="********"
              class="w-full h-11 px-3 py-3 rounded-md border border-blue-gray-200 focus:border-gray-900"
              autocomplete="new-password"
              required
          />
        </div>
        <button
            type="submit"
            class="w-full py-3.5 px-7 rounded-lg bg-gray-900 text-white shadow-md hover:shadow-lg mt-6"
        >
          注册
        </button>
      </form>
    </div>
  </section>
</template>

<script>
import Header from "@/components/Header.vue";
import showToast from "@/utils/message.js";
import axios from "axios";


export default {
  components: {Header},
  data() {
    return {
      username: '',
      password: '',
      re_password: ''
    };
  },
  methods: {
    register() {
      const username = this.username
      const password = this.password
      const re_password = this.re_password
      if (password !== re_password) {
        showToast('error', '两次输入密码不一致')
        return
      }
      const formData = {
        'username': username,
        'password': password
      }
      axios
          .post('/auth/users/', formData)
          .then(response => {
            showToast('success', '注册成功',()=>{
              this.$router.push({name:'login'})
            })
          })
          .catch(error => {
            const errorData = error.response.data
            const errorMessage = Object.values(errorData).flat()
            for (let i=0; i<errorMessage.length;i++) {
              showToast('error',errorMessage[i])
            }
          })
    }
  }
};
</script>
