<template>
  <div
      class="rounded-2xl border border-gray-200 bg-gray-100 dark:border-gray-800 dark:bg-white/[0.03]"
  >
    <div
        class="px-5 pt-5 bg-white shadow-default rounded-2xl pb-11 dark:bg-gray-900 sm:px-6 sm:pt-6"
    >
      <div class="flex justify-between">
        <div>
          <h3 class="text-lg font-semibold text-gray-800 dark:text-white/90">数据获取进度</h3>
          <p class="mt-1 text-gray-500 text-theme-sm dark:text-gray-400">
            获取各站点降水及水位数据及各区域气象信息
          </p>
        </div>
        <div>
          <DropdownMenu :menu-items="menuItems">
            <template #icon>
              <svg
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
              >
                <path
                    fillRule="evenodd"
                    clipRule="evenodd"
                    d="M10.2441 6C10.2441 5.0335 11.0276 4.25 11.9941 4.25H12.0041C12.9706 4.25 13.7541 5.0335 13.7541 6C13.7541 6.9665 12.9706 7.75 12.0041 7.75H11.9941C11.0276 7.75 10.2441 6.9665 10.2441 6ZM10.2441 18C10.2441 17.0335 11.0276 16.25 11.9941 16.25H12.0041C12.9706 16.25 13.7541 17.0335 13.7541 18C13.7541 18.9665 12.9706 19.75 12.0041 19.75H11.9941C11.0276 19.75 10.2441 18.9665 10.2441 18ZM11.9941 10.25C11.0276 10.25 10.2441 11.0335 10.2441 12C10.2441 12.9665 11.0276 13.75 11.9941 13.75H12.0041C12.9706 13.75 13.7541 12.9665 13.7541 12C13.7541 11.0335 12.9706 10.25 12.0041 10.25H11.9941Z"
                    fill="currentColor"
                />
              </svg>
            </template>
          </DropdownMenu>
        </div>
      </div>
      <div class="relative max-h-[195px]">
        <div id="chartTwo" class="h-full">
          <div class="radial-bar-chart">
            <VueApexCharts type="radialBar" height="330" :options="chartOptions" :series="series"/>
          </div>
        </div>
        <span
            class="absolute left-1/2 top-[85%] -translate-x-1/2 -translate-y-[85%] rounded-full bg-success-50 px-3 py-1 text-xs font-medium text-success-600 dark:bg-success-500/15 dark:text-success-500"
        >+10%</span
        >
      </div>
      <p class="mx-auto mt-1.5 w-full max-w-[380px] text-center text-sm text-gray-500 sm:text-base">
        You earn $3287 today, it's higher than last month. Keep up your good work!
      </p>
    </div>

    <div class="flex items-center justify-center gap-5 px-6 py-3.5 sm:gap-8 sm:py-5">
      <div>
        <p class="mb-1 text-center text-gray-500 text-theme-xs dark:text-gray-400 sm:text-sm">
          Target
        </p>
        <p
            class="flex items-center justify-center gap-1 text-base font-semibold text-gray-800 dark:text-white/90 sm:text-lg"
        >
          $20K
          <svg
              width="16"
              height="16"
              viewBox="0 0 16 16"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
          >
            <circle cx="7" cy="7" r="6" stroke="#12B76A" stroke-width="2" fill="none"/>
          </svg>
          <!-- 空心绿色圆形图标，边框颜色为 #12B76A -->

        </p>
      </div>

      <div class="w-px bg-gray-200 h-7 dark:bg-gray-800"></div>

      <div>
        <p class="mb-1 text-center text-gray-500 text-theme-xs dark:text-gray-400 sm:text-sm">
          Revenue
        </p>
        <p
            class="flex items-center justify-center gap-1 text-base font-semibold text-gray-800 dark:text-white/90 sm:text-lg"
        >
          $20K
          <svg
              width="16"
              height="16"
              viewBox="0 0 16 16"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
          >
            <path
                fill="none"
                stroke="#12B76A"
                stroke-width="2"
                d="M4 8L7 11L12 4"
            />
          </svg>
          <!-- 绿色勾图标，勾的边框颜色为 #12B76A -->



        </p>
      </div>

      <div class="w-px bg-gray-200 h-7 dark:bg-gray-800"></div>

      <div>
        <p class="mb-1 text-center text-gray-500 text-theme-xs dark:text-gray-400 sm:text-sm">
          Today
        </p>
        <p
            class="flex items-center justify-center gap-1 text-base font-semibold text-gray-800 dark:text-white/90 sm:text-lg"
        >
          $20K
          <!-- 错误 -->
          <svg
              width="16"
              height="16"
              viewBox="0 0 16 16"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
          >
            <path
                fill-rule="evenodd"
                clip-rule="evenodd"
                d="M8 6.93934L11.4697 3.46967C11.7626 3.17678 12.2374 3.17678 12.5303 3.46967C12.8232 3.76256 12.8232 4.23744 12.5303 4.53033L9.06066 8L12.5303 11.4697C12.8232 11.7626 12.8232 12.2374 12.5303 12.5303C12.2374 12.8232 11.7626 12.8232 11.4697 12.5303L8 9.06066L4.53033 12.5303C4.23744 12.8232 3.76256 12.8232 3.46967 12.5303C3.17678 12.2374 3.17678 11.7626 3.46967 11.4697L6.93934 8L3.46967 4.53033C3.17678 4.23744 3.17678 3.76256 3.46967 3.46967C3.76256 3.17678 4.23744 3.17678 4.53033 3.46967L8 6.93934Z"
                fill="#D92D20"
            />
          </svg>
        </p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import {ref, computed} from 'vue'
import DropdownMenu from '../common/DropdownMenu.vue'

const menuItems = [
  {label: 'View More', onClick: () => console.log('View More clicked')},
  {label: 'Delete', onClick: () => console.log('Delete clicked')},
]
import VueApexCharts from 'vue3-apexcharts'

const props = defineProps({
  value: {
    type: Number,
    default: 0,
  },
})

const series = computed(() => [props.value])

const chartOptions = {
  colors: ['#465FFF'],
  chart: {
    fontFamily: 'Outfit, sans-serif',
    sparkline: {
      enabled: true,
    },
  },
  plotOptions: {
    radialBar: {
      startAngle: -90,
      endAngle: 90,
      hollow: {
        size: '80%',
      },
      track: {
        background: '#E4E7EC',
        strokeWidth: '100%',
        margin: 5,
      },
      dataLabels: {
        name: {
          show: false,
        },
        value: {
          fontSize: '36px',
          fontWeight: '600',
          offsetY: 60,
          color: '#1D2939',
          formatter: function (val: number) {
            return val.toFixed(2) + '%'
          },
        },
      },
    },
  },
  fill: {
    type: 'solid',
    colors: ['#465FFF'],
  },
  stroke: {
    lineCap: 'round',
  },
  labels: ['Progress'],
}
</script>

<style scoped>
.radial-bar-chart {
  width: 100%;
  max-width: 330px;
  margin: 0 auto;
}
</style>
