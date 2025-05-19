<template>
  <div
    class="rounded-2xl border border-gray-200 bg-white p-6 dark:border-gray-800 dark:bg-white/[0.03]"
  >
    <div class="mb-6 flex justify-between">
      <div>
        <h3 class="text-lg font-semibold text-gray-800 dark:text-white/90">其他</h3>
      </div>
    </div>
    <div class="flex w-full items-center gap-0.5 rounded-lg bg-gray-100 p-0.5 dark:bg-gray-900">
      <button
        @click="selected = 'rain'"
        :class="[
          'text-sm w-full rounded-md px-3 py-2 font-medium hover:text-gray-900 dark:hover:text-white',
          selected === 'rain'
            ? 'shadow-sm text-gray-900 dark:text-white bg-white dark:bg-gray-800'
            : 'text-gray-500 dark:text-gray-400',
        ]"
      >
        降水量
      </button>
      <button
        @click="selected = 'pred'"
        :class="[
          'text-sm w-full rounded-md px-3 py-2 font-medium hover:text-gray-900 dark:hover:text-white',
          selected === 'pred'
            ? 'shadow-sm text-gray-900 dark:text-white bg-white dark:bg-gray-800'
            : 'text-gray-500 dark:text-gray-400',
        ]"
      >
        6小时预测
      </button>
      <button
        @click="selected = 'warning'"
        :class="[
          'text-sm w-full rounded-md px-3 py-2 font-medium hover:text-gray-900 dark:hover:text-white',
          selected === 'warning'
            ? 'shadow-sm text-gray-900 dark:text-white bg-white dark:bg-gray-800'
            : 'text-gray-500 dark:text-gray-400',
        ]"
      >
        告警处理
      </button>
    </div>

    <!-- Tab Panels -->
    <div class="mt-4">
      <!-- Daily Sales Panel -->
      <div v-if="selected === 'rain'" class="space-y-4">
        <div
          class="grid grid-cols-2 justify-between gap-10 divide-x divide-gray-100 rounded-xl border border-gray-100 bg-white py-4 dark:divide-gray-800 dark:border-gray-800 dark:bg-gray-800/[0.03]"
        >
          <div class="px-5">
            <span class="block text-sm text-gray-500 dark:text-gray-400"> 近一月降水量 </span>
            <div class="mt-1 flex items-center gap-2">
              <h4 class="text-xl font-semibold text-gray-800 dark:text-white/90">{{ rainData[0].data.reduce((sum, item) => sum + item.y, 0).toFixed(2) + 'mm'}}</h4>
            </div>
          </div>
          <div class="px-5">
            <span class="block text-sm text-gray-500 dark:text-gray-400"> 近24小时降水 </span>
            <div class="mt-1 flex items-center gap-2">
              <h4 class="text-xl font-semibold text-gray-800 dark:text-white/90">
                {{ rainData[0].data.slice(-24).reduce((sum, item) => sum + item.y, 0).toFixed(2) + 'mm' }}
              </h4>
            </div>
          </div>
        </div>
        <div class="rounded-xl border border-gray-100 px-5 py-4 dark:border-gray-800">
          <!-- Chart placeholder -->
          <div class="overflow-x-auto">
            <RecentRainChart :series="rainData"/>
          </div>
        </div>
      </div>

      <!-- Online Sales Panel -->
      <div v-if="selected === 'pred'" class="space-y-4">
        <div class="rounded-xl border border-gray-100 px-5 py-4 dark:border-gray-800">
          <div class="mb-3 flex items-start justify-between">
            <div>
              <span class="text-sm font-medium text-gray-500 dark:text-gray-400">
                未来6小时后水位
              </span>
              <h3 v-if='predData[0].data.length >0'
                  class="text-2xl font-semibold text-gray-800 dark:text-white/90">
                {{ predData[0]?.data?.at(-1).y }}
              </h3>
            </div>
          </div>
          <!-- Chart placeholder -->
          <div v-if='predData[0].data.length >0' class="custom-scrollbar max-w-full overflow-x-auto">
            <WaterPredChart :series="predData" />
          </div>
          <div v-else class="custom-scrollbar max-w-full overflow-x-auto">
            <h3 class="text-xl font-semibold text-gray-800 dark:text-white/90">数据缺失/该站点无可用预测模型</h3>
          </div>
        </div>
      </div>

      <!-- New Users Panel -->
      <div v-if="selected === 'warning'" class="space-y-4">
        <div class="rounded-xl border border-gray-100 px-5 py-4 dark:border-gray-800">
          <WarnCancelFrom v-if="status !== 0" :station_id="String(station_id)"/>
          <h3 v-else class="text-xl font-semibold text-gray-800 dark:text-white/90">站点正常</h3>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import {defineProps, ref} from 'vue'
import RecentRainChart from './RecentRainChart.vue'
import WarnCancelFrom from './WarnCancelFrom.vue'
import WaterPredChart from './WaterPredChart.vue'

const {rainData, predData, status, station_id} = defineProps({
  rainData: {
    type: Object,
    required: true
  },
  predData: {
    type: Object,
    required: true
  },
  status: {
    type: Number,
    required: true
  },
  station_id: {
    type: String,
    required: true
  }
})

const selected = ref('rain')

</script>
