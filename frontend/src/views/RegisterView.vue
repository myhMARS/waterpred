<template>
  <Header/>
  <section class="grid text-center h-screen items-center p-8">
    <div>
      <h3 class="text-3xl font-semibold text-blue-gray-900 mb-2">注 册</h3>

      <form @submit.prevent="handleSubmit" class="mx-auto max-w-[24rem] text-left">
        <div class="mb-6">
          <label for="username" class="block font-medium text-gray-900 mb-2">用户名</label>
          <input
              id="username"
              type="text"
              v-model="username"
              placeholder="Enter your username"
              class="w-full h-11 px-3 py-3 rounded-md border border-blue-gray-200 focus:border-gray-900"
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
              required
          />
        </div>
        <div class="mb-6">
          <label for="password" class="block font-medium text-gray-900 mb-2">再次输入密码</label>
          <input
              id="password"
              type="password"
              v-model="password"
              placeholder="********"
              class="w-full h-11 px-3 py-3 rounded-md border border-blue-gray-200 focus:border-gray-900"
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

export default {
  components: {Header},
  data() {
    return {
      username: '',
      password: ''
    };
  },
  methods: {
    handleSubmit() {
      console.log("Username:", this.username);
      console.log("Password:", this.password);
      alert("Login submitted");
    }
  }
};
</script>
