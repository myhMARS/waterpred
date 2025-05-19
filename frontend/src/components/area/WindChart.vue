<template>
  <div
      class="rounded-2xl border border-gray-200 bg-white px-5 pb-5 pt-5 dark:border-gray-800 dark:bg-white/[0.03] sm:px-6 sm:pt-6"
  >
    <div class="flex flex-col gap-4 mb-6 sm:flex-row sm:items-center sm:justify-between">
      <div>
        <h3 class="text-lg font-semibold text-gray-800 dark:text-white/90">
          风向玫瑰图
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
            height="310"
            type="radar"
            :options="chartOptions"
            :series="series"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import VueApexCharts from 'vue3-apexcharts'
import {ref} from "vue";

const {data} = defineProps({
  data: {
    type: Object,
    required: true
  }
})
const series = ref([{
  name: '风频率',
  data: data.winddirection
}]); // 每个方向的风频率或风速

const chartOptions = ref({
  labels: ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
  chart: {
    type: 'radar'
  },
  stroke: {
    width: 2
  },
  fill: {
    opacity: 0.25
  },
  markers: {
    size: 4
  },
  yaxis: {
    show: false
  },
  xaxis: {
    labels: {
      show: true
    }
  }
});

</script>
