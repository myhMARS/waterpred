<template>
  <div>
    <VueApexCharts
        class="-ml-3"
        :options="chartOptions"
        :series="series"
        type="area"
        height="200"
    ></VueApexCharts>
  </div>
</template>

<script setup>
import {ref, onMounted} from 'vue'
import VueApexCharts from 'vue3-apexcharts'

const chartOptions = ref({
  colors: ['#465FFF', '#9CB9FF'],
  chart: {
    fontFamily: 'Outfit, sans-serif',
    type: 'area',
    toolbar: {
      show: false,
    },
    zoom: {
      enabled: true,
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
    x: {
      format: 'yyyy/MM/dd HH:mm:ss'
    }
  },
  yaxis: {
    title: {
      style: {
        fontSize: '0px',
      },
    },
  },
})

const {series} = defineProps({
  series: {
    type: Object,
    required: true
  }
})
</script>
