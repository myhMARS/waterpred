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
import { ref } from 'vue'
import VueApexCharts from 'vue3-apexcharts'

const chartOptions = ref({
  legend: {
    show: false,
    position: 'top',
    horizontalAlign: 'left',
  },
  colors: ['#465FFF', '#9CB9FF'],
  chart: {
    fontFamily: 'Outfit, sans-serif',
    height: 310,
    type: 'area',
    toolbar: {
      show: false,
    },
    zoom: {
      enabled: true,
      allowMouseWheelZoom: false, // Disable mouse wheel zooming
    },
  },
  fill: {
    type: 'gradient', // Explicitly specify gradient type
    gradient: {
      opacityFrom: 0.55,
      opacityTo: 0,
    },
  },
  stroke: {
    curve: 'smooth',
    width: [2, 2], // Correct width as an array of numbers
  },
  markers: {
    size: 0,
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
  tooltip: {
    x: {
      format: 'yyyy/MM/dd HH:mm:ss',
    },
  },
  xaxis: {
    type: 'datetime',
    labels: {
      datetimeUTC: false
    },
    axisBorder: {
      show: false,
    },
    axisTicks: {
      show: false,
    },
    tooltip: {
      enabled: false, // Correct usage for disabling tooltips
    },
  },
  yaxis: {
    title: {
      text: '',
      style: {
        fontSize: '0px',
      },
    },
  },
})

const { series } = defineProps({
  series: {
    type: Object,
    required: true
  }
})
</script>
