<template>
  <div>
    <div class="p-5 mb-6 border border-gray-200 rounded-2xl dark:border-gray-800 lg:p-6">
      <div class="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h4 class="text-lg font-semibold text-gray-800 dark:text-white/90 lg:mb-6">
            个人信息
          </h4>

          <div class="grid grid-cols-1 gap-4 lg:grid-cols-2 lg:gap-7 2xl:gap-x-32">
            <div>
              <p class="mb-2 text-xs leading-normal text-gray-500 dark:text-gray-400">姓名</p>
              <p class="text-sm font-medium text-gray-800 dark:text-white/90">{{ username }}</p>
            </div>

            <div>
              <p class="mb-2 text-xs leading-normal text-gray-500 dark:text-gray-400">
                邮件地址
              </p>
              <p class="text-sm font-medium text-gray-800 dark:text-white/90">
                {{ email }}
              </p>
            </div>

            <div>
              <p class="mb-2 text-xs leading-normal text-gray-500 dark:text-gray-400">手机号</p>
              <p class="text-sm font-medium text-gray-800 dark:text-white/90">{{ phonenumber }}</p>
            </div>

            <div>
              <p class="mb-2 text-xs leading-normal text-gray-500 dark:text-gray-400">职务</p>
              <p class="text-sm font-medium text-gray-800 dark:text-white/90">{{ manager }}</p>
            </div>
          </div>
        </div>


      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const username = ref('')
const email = ref('')
const phonenumber = ref('')
const manager = ref('')

onMounted(() => {
  username.value = localStorage.getItem('username')
  email.value = localStorage.getItem('email')
  phonenumber.value = localStorage.getItem('phone')
  manager.value = localStorage.getItem('manager') || '暂无'
})

</script>
