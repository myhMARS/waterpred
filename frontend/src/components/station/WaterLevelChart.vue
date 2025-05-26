<template>
  <div
      class="rounded-2xl border border-gray-200 bg-white px-5 pb-5 pt-5 dark:border-gray-800 dark:bg-white/[0.03] sm:px-6 sm:pt-6"
  >
    <div class="flex flex-col gap-4 mb-6 sm:flex-row sm:items-center sm:justify-between">
      <div>
        <h3 class="text-lg font-semibold text-gray-800 dark:text-white/90">
          站点水位
        </h3>
        <p class="mt-1 text-gray-500 text-theme-sm dark:text-gray-400">近30天</p>
      </div>

      <div
          class="flex flex-row-reverse items-center justify-end gap-0.5 sm:flex-col sm:items-start"
      >
        <div class="flex flex-row-reverse items-center gap-3 sm:flex-row sm:gap-2">
          <h4 class="text-2xl font-bold text-gray-800 dark:text-white/90">{{ series[0]?.data?.length ? series[0].data.at(-1).y + ' m' : '暂无数据' }}</h4>

        </div>

      </div>
    </div>
    <div class="max-w-full overflow-x-auto custom-scrollbar">
      <div
          id="chartEight"
          class="apexcharts-tooltip-active -ml-4 min-w-[1000px] pl-2 xl:min-w-full"
      >
        <VueApexCharts v-if="series"
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
import {ref, onMounted, defineProps} from 'vue'
import VueApexCharts from 'vue3-apexcharts'

const { series } = defineProps({
  series: {
    type: Object,
    required: true
  },
})
const chartOptions = ref({
  colors: ['#465FFF', '#ef4444', '#facc15', '#ef4444'],
  chart: {
    fontFamily: 'Outfit, sans-serif',
    type: 'area',
    toolbar: {
      show: false,
    },
  },
  legend: {
    show: false,
    position: 'top',
    horizontalAlign: 'left',
  },
  fill: {
    gradient: {
      enabled: true,
      opacityFrom: 0.55,
      opacityTo: 0,
    },
  },
  stroke: {
    curve: 'smooth',
    width: [2, 2],
  },
  markers: {
    size: 0,
  },
  labels: {
    show: false,
    position: 'top',
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
  dataLabels: {
    enabled: false,
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
  tooltip: {
    shared: true,
    intersect: false,
    x: {
      format: 'yyyy/MM/dd HH:mm:ss'
    },
  },
  yaxis: {
    title: {
      style: {
        fontSize: '0px',
      },
    },
  },
})

</script>

<style></style>
