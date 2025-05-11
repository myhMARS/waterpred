<template>
  <div>
    <div class="p-5 mb-6 border border-gray-200 rounded-2xl dark:border-gray-800 lg:p-6">
      <div class="flex flex-col gap-5 xl:flex-row xl:items-center xl:justify-between">
        <div class="flex flex-col items-center w-full gap-6 xl:flex-row">
          <div class="order-3 xl:order-2">
            <h4
                class="mb-2 text-lg font-semibold text-center text-gray-800 dark:text-white/90 xl:text-left"
            >
              {{ username }}
            </h4>
            <div
                class="flex flex-col items-center gap-1 text-center xl:flex-row xl:gap-3 xl:text-left"
            >
              <p class="text-sm text-gray-500 dark:text-gray-400">工号: {{ userid }}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import {ref} from "vue";

const username = ref('')
const manager = ref('')
const location = ref('')
const userid = ref('')
username.value = localStorage.getItem('username')
manager.value = localStorage.getItem('manager') || '暂无'
location.value = localStorage.getItem('location') || '暂无'
userid.value = localStorage.getItem('userid')
</script>
