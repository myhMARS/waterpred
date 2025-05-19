<template>
  <div
      class="rounded-2xl border border-gray-200 bg-white px-5 pb-5 pt-5 dark:border-gray-800 dark:bg-white/[0.03] sm:px-6 sm:pt-6"
  >
    <div class="flex flex-col gap-4 mb-6 sm:flex-row sm:items-center sm:justify-between">
      <div>
        <h3 class="text-lg font-semibold text-gray-800 dark:text-white/90">
          风速气温湿度数据图
        </h3>
        <p class="mt-1 text-gray-500 text-theme-sm dark:text-gray-400">{{ data.startDate }} - {{ data.endDate }}</p>
      </div>
    </div>
    <div class="max-w-full overflow-x-auto custom-scrollbar">
      <div
          id="chartEight"
          class="apexcharts-tooltip-active -ml-4 pl-2 min-w-full"
      >
        <VueApexCharts
            type="area"
            height="310"
            :options="chartOptions"
            :series="series"
        ></VueApexCharts>
      </div>
    </div>
  </div>
</template>

<script setup lang="js">
import {ref, onMounted, onUnmounted} from 'vue'
import VueApexCharts from 'vue3-apexcharts'

const {data} = defineProps({
  data: {
    type: Object,
    required: true
  }
})

const series = ref([
  {
    name: '风速 (m/s)',
    type: 'line',
    data: data.windpower
  },
  {
    name: '气温 (°C)',
    type: 'line',
    data: data.temperature
  },
  {
    name: '湿度 (%)',
    type: 'line',
    data: data.humidity
  }
])

const chartOptions = ref({
  chart: {
    height: 310,
    type: 'line',
    toolbar: {
      show: false,
    },
  },
  labels: {
    show: false,
    position: 'top',
  },
  legend: {
    show: false,
    position: 'top',
    horizontalAlign: 'left',
  },
  dataLabels: {
    enabled: false
  },
  grid: {
    xaxis: {
      lines: {
        show: false,
      },
    },
    yaxis: {
      lines: {
        show: true,
      },
    },
  },
  fill: {
    type: 'solid',     // 必须设置为 solid，避免透明渐变
    opacity: 1         // 设置不透明度为 1（完全不透明）
  },
  stroke: {
    curve: 'smooth',
    width: [2, 2, 2],
  },
  title: {
    text: '风速、气温与湿度'
  },
  xaxis: {
    type: 'datetime',
    labels: {
      show: false,
      datetimeUTC: false
    },
    axisBorder: {
      show: false,
    },
    axisTicks: {
      show: false,
    },
    tooltip: {
      enabled: false,
    },
  },
  yaxis: [
    {
      title: {
        text: '风速 (m/s)'
      },
      opposite: false
    },
    {
      title: {
        text: '气温 (°C)'
      },
      opposite: false
    },
    {
      title: {
        text: '湿度 (%)'
      },
      opposite: false
    }
  ],
  tooltip: {
    x: {
      format: 'yyyy/MM/dd HH:mm:ss'
    }
  }
})
</script>

<style></style>
