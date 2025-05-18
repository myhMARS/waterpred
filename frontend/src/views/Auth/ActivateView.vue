<template>
  <FullScreenLayout>
    <div class="relative p-6 bg-white z-1 dark:bg-gray-900 sm:p-0">
      <div
          class="relative flex lg:flex-row w-full h-screen justify-center flex-col dark:bg-gray-900"
      >
        <div class="flex flex-col flex-1 lg:w-1/2 w-full">
          <div class="w-full max-w-md pt-10 mx-auto">
            <router-link
                to="/"
                class="inline-flex items-center text-sm text-gray-500 transition-colors hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
            >
              <svg
                  class="stroke-current"
                  xmlns="http://www.w3.org/2000/svg"
                  width="20"
                  height="20"
                  viewBox="0 0 20 20"
                  fill="none"
              >
                <path
                    d="M12.7083 5L7.5 10.2083L12.7083 15.4167"
                    stroke=""
                    stroke-width="1.5"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                />
              </svg>
              Back to home
            </router-link>
          </div>
          <!-- Form -->
          <div class="flex flex-col justify-center flex-1 w-full max-w-md mx-auto">
            <div class="mb-5 sm:mb-8">
              <h1
                  class="mb-2 font-semibold text-gray-800 text-title-sm dark:text-white/90 sm:text-title-md"
              >
                邮箱激活
              </h1>
              <p class="text-sm text-gray-500 dark:text-gray-400">
                本链接仅用于激活您注册的账户。若非本人操作，请勿点击。
              </p>
            </div>
            <div>

              <!-- Button -->
              <div>
                <button
                    v-on:click='activate'
                    class="flex items-center justify-center w-full px-4 py-3 text-sm font-medium text-white transition rounded-lg bg-brand-500 shadow-theme-xs hover:bg-brand-600"
                >
                  激活我的账号
                </button>
              </div>
            </div>
          </div>
        </div>
        <div
            class="lg:w-1/2 w-full h-full bg-brand-950 dark:bg-white/5 lg:grid items-center hidden relative"
        >
          <div class="items-center justify-center flex z-1">
            <common-grid-shape/>
            <div class="flex flex-col items-center max-w-xs">
              <router-link to="/" class="block mb-4">
                <img width="{231}" height="{48}" src="/images/logo/auth-logo.svg" alt="Logo"/>
              </router-link>
            </div>
          </div>
        </div>
      </div>
    </div>
  </FullScreenLayout>
</template>

<script setup lang="js">
import CommonGridShape from '@/components/common/CommonGridShape.vue'
import FullScreenLayout from '@/components/layout/FullScreenLayout.vue'
import axios from "axios"
import {showMessage} from "@/utils/vuemessage.js";
import { useRoute, useRouter } from 'vue-router'

const route = useRoute()
const router = useRouter()

function activate() {
  const formData = {
    uid: route.params.uid,
    token: route.params.token
  }
  axios.post('/auth/users/activation/', formData)
      .then(response => {
        showMessage({
          type: 'success',
          message: '激活成功',
          onClose: () => {
            router.push({name: 'login'})
          }
        })
      })
      .catch(error => {
        const errorData = error.response.data
        const errorMessage = Object.values(errorData).flat()
        for (let i = 0; i < errorMessage.length; i++) {
          showMessage({
            type: 'error',
            message: errorMessage[i]
          })
        }
      })
}
</script>
