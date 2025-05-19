<template>
  <div
    class="rounded-2xl border border-gray-200 bg-white p-5 dark:border-gray-800 dark:bg-white/[0.03]"
  >
    <div class="mb-8 flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
      <div>
        <h3 class="text-lg font-semibold text-gray-800 dark:text-white">站点数据</h3>
      </div>
    </div>
    <div
      class="grid rounded-2xl border border-gray-200 bg-white sm:grid-cols-2 xl:grid-cols-4 dark:border-gray-800 dark:bg-gray-900"
    >
      <div
        class="border-b border-gray-200 px-6 py-5 sm:border-r xl:border-b-0 dark:border-gray-800"
      >
        <span class="text-sm text-gray-500 dark:text-gray-400">上报时间</span>
        <div class="mt-2 flex items-end gap-3">
          <h6 class="text-title-xs sm:text-title-sm font-bold text-gray-800 dark:text-white/90">
            {{ data.time===null?'暂无': data.time }}
          </h6>
        </div>
      </div>
      <div
        class="border-b border-gray-200 px-6 py-5 xl:border-r xl:border-b-0 dark:border-gray-800"
      >
        <span class="text-sm text-gray-500 dark:text-gray-400">站点降水量</span>
        <div class="mt-2 flex items-end gap-3">
          <h4 class="text-title-xs sm:text-title-sm font-bold text-gray-800 dark:text-white/90">
            {{ data.rain===null? '暂无': data.rain + 'mm'}}
          </h4>
        </div>
      </div>
      <div
        class="border-b border-gray-200 px-6 py-5 sm:border-r sm:border-b-0 dark:border-gray-800"
      >
        <div>
          <span class="text-sm text-gray-500 dark:text-gray-400">站点水位</span>
          <div class="mt-2 flex items-end gap-3">
            <h4 class="text-title-xs sm:text-title-sm font-bold text-gray-800 dark:text-white/90">
              {{ data.waterlevel===null? '暂无':data.waterlevel + 'm' }}
            </h4>
          </div>
        </div>
      </div>
      <div class="px-6 py-5">
        <span class="text-sm text-gray-500 dark:text-gray-400">站点状态</span>
        <div class="mt-2 flex items-end gap-3">
          <h4 class="text-title-xs sm:text-title-sm font-bold"
          :class="{
            'text-success-600 dark:text-success-500': data.status === 0,
            'text-error-600 dark:text-error-500': data.status !== 0
            }">
            {{ data.status===0?'正常':'告警中' }}
          </h4>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { defineProps } from 'vue'

defineProps({
  data: {
    type: Object,
    required: true
  }
})
</script>

