<template>
  <div
      class="overflow-hidden rounded-2xl border border-gray-200 bg-white p-6 dark:border-gray-800 dark:bg-white/[0.03]"
  >
    <div class="mb-6 flex justify-between">
      <div>
        <h3 class="text-lg font-semibold text-gray-800 dark:text-white/90">水位变化</h3>
        <p class="text-sm mt-1 text-gray-500 dark:text-gray-400">与12小时前相比</p>
      </div>
    </div>
    <div class="flex justify-between">
      <div v-if="series[0].data.length !== 0">
        <h3 class="text-xl font-semibold text-gray-800 dark:text-white/90">{{ series[0].data.at(-1) }} m</h3>
        <p class="text-xs mt-1 text-gray-500 dark:text-gray-400">
          <p v-if='series[0].data.at(-1) !== null' class="text-xs mt-1 text-gray-500 dark:text-gray-400">
          <span :class="{
            'text-red-500 mr-1 inline-block':diff>0,
            'text-green-500 mr-1 inline-block':diff<=0,
          }">
            {{ diff > 0 ? '+' + diff : '' + diff }} m
          </span>
            相较12小时前
          </p>
        </p>
      </div>
      <div v-if="series[0].data.length === 0">
        <h3 class="text-xl font-semibold text-gray-800 dark:text-white/90">暂无数据</h3>
      </div>
      <div class="max-w-full">
        <div id="chartTwentyTwo">
          <VueApexCharts
              class="h-12 w-24"
              :options="chartOptions"
              :series="series"
              type="area"
          ></VueApexCharts>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import {onMounted, ref, computed} from 'vue'
import VueApexCharts from 'vue3-apexcharts'
import axios from "axios";

const diff = ref()
const series = ref([
  {
    name: '水位',
    data: [],
  },
])
const {stationid} = defineProps({
  stationid: {
    type: String,
    required: true
  }
})

async function fetchData() {
  try {
    const response = await axios.get('/api/recent/', {
      params: {
        stationid: stationid,
        waterlevel: true
      },
      headers: {
        Authorization: `JWT ${localStorage.getItem('token')}`
      }
    })
    series.value[0].data = response.data
    diff.value = parseFloat((series.value[0].data.at(-1) - series.value[0].data.at(0)).toFixed(2))
  } catch (error) {
    console.log(error)
  }
}

onMounted(() => {
  fetchData()
})

const chartOptions = computed(() => ({
      chart: {
        type: 'area',
        height: 60,
        sparkline: {
          enabled: true,
        },
        animations: {
          enabled: true,
          speed: 800,
        },
        toolbar: {
          show: false,
        },
      },
      colors: diff.value > 0 ? ['#ef4444'] : ['#10b981'],
      stroke: {
        curve: 'smooth',
        width: 2,
      },
      fill: {
        type: 'gradient',
        gradient: {
          shadeIntensity: 1,
          opacityFrom: 0.6,
          opacityTo: 0.1,
          stops: [0, 100],
        },
      },
      tooltip: {
        fixed: {
          enabled: false,
        },
        x: {
          show: false,
        },
        y: {
          formatter: (value) => value.toLocaleString() + 'm',
        },
        marker: {
          show: false,
        },
      },
    })
)


</script>
