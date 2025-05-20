<template>
  <div
      class="rounded-2xl border border-gray-200 bg-white px-5 pb-5 pt-5 dark:border-gray-800 dark:bg-white/[0.03] sm:px-6 sm:pt-6"
  >
    <div class="flex flex-col gap-5 mb-6 sm:flex-row sm:justify-between">
      <div class="w-full">
        <h3 class="text-lg font-semibold text-gray-800 dark:text-white/90">统计</h3>
      </div>

      <div class="relative">
        <div class="inline-flex items-center gap-0.5 rounded-lg bg-gray-100 p-0.5 dark:bg-gray-900">
          <button
              v-for="option in options"
              :key="option.value"
              @click="chageChart(option.value)"
              :class="[
              selected === option.value
                ? 'shadow-theme-xs text-gray-900 dark:text-white bg-white dark:bg-gray-800'
                : 'text-gray-500 dark:text-gray-400',
              'px-4 py-2 font-medium whitespace-nowrap rounded-md text-theme-sm hover:text-gray-900 hover:shadow-theme-xs dark:hover:bg-gray-800 dark:hover:text-white',
            ]"
          >
            {{ option.label }}
          </button>
        </div>
      </div>
    </div>
    <div class="w-full overflow-x-hidden">
      <div id="chartThree" class="pl-2">
        <VueApexCharts type="area" height="310" v-if="chartOptions" :options="chartOptions" :series="series"/>
      </div>
    </div>
  </div>
</template>

<script setup>
import {onMounted, ref} from 'vue'
import VueApexCharts from 'vue3-apexcharts'
import axios from "axios";

const options = [
  {value: 'month', label: '月度'},
  {value: 'quarter', label: '季度'},
  {value: 'year', label: '年度'},
]

const selected = ref('month')

const series = ref()
const timelineStart = ref(null)
const timelineEnd = ref(null)
const cacheData = ref({
  monthData: {},
  quarterData: {},
  yearData: {}
})

function chageChartOption(startTime, endTime, timelength) {
  chartOptions.value = {
    legend: {
      show: false,
      position: 'top',
      horizontalAlign: 'left',
    },

    colors: ['#465FFF', '#9CB9FF'],
    chart: {
      fontFamily: 'Outfit, sans-serif',
      type: 'area',
      toolbar: {
        show: false,
      },
      zoom: {
        enabled: false,
      },
    },
    fill: {
      gradient: {
        enabled: true,
        opacityFrom: 0.55,
        opacityTo: 0,
      },
    },
    stroke: {
      curve: 'straight',
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
    tooltip: {
      x: {
        format: 'dd MMM yyyy',
      },
    },
    xaxis: {
      type: 'datetime',
      min: startTime,
      max: endTime,
      tickAmount: timelength,
      axisBorder: {
        show: false,
      },
      axisTicks: {
        show: false,
      },
      labels: {
        formatter: function (value) {
          const date = new Date(value);
          const options = {year: 'numeric', month: 'long', day: 'numeric'};
          return new Intl.DateTimeFormat('zh-CN', options).format(date);
        }
      },
      tooltip: {
        enabled: false,
      },
    },
    yaxis: {
      labels: {
        formatter: function (val) {
          return val.toFixed(0); // 强制显示整数
        }
      },
      title: {
        style: {
          fontSize: '0px',
        },
      },
    },
  }

}

async function fetchdata() {
  await axios.get('/api/statistics/', {
    params: {
      time_filter: selected.value
    },
    headers: {
      Authorization: `JWT ${localStorage.getItem('token')}`
    }
  })
      .then(response => {
        const rawData = response.data.data
        const year = response.data.year
        const month = response.data.month
        const days = response.data.day
        const millisecondsPerDay = 24 * 60 * 60 * 1000;
        if (selected.value === 'month') {
          timelineStart.value = new Date(year, month - 1, 1).getTime()
          // 获取这个月有多少天
          const daysInMonth = new Date(year, month, 0).getDate();
          timelineEnd.value = new Date(year, month - 1, daysInMonth).getTime()
          chageChartOption(timelineStart.value, timelineEnd.value, daysInMonth - 1)
          cacheData.value.monthData = {
            startTime: timelineStart.value,
            endTime: timelineEnd.value,
            tick: daysInMonth,
          }
        } else if (selected.value === 'quarter') {
          const quarter = Math.floor((month - 1) / 3) + 1
          timelineStart.value = new Date(year, (quarter - 1) * 3, 1).getTime()
          const daysInMonth = new Date(year, (quarter - 1) * 3 + 2, 0).getDate()
          timelineEnd.value = new Date(year, (quarter - 1) * 3 + 2, daysInMonth).getTime()
          // 1天 = 86400000 毫秒
          chageChartOption(timelineStart.value, timelineEnd.value, 12)
          cacheData.value.quarterData = {
            startTime: timelineStart.value,
            endTime: timelineEnd.value,
            tick: 12,
          }
        } else if (selected.value === 'year') {
          timelineStart.value = new Date(year, 0, 1).getTime()
          timelineEnd.value = new Date(year, 11, 31).getTime()
          chageChartOption(timelineStart.value, timelineEnd.value, 12)
          cacheData.value.yearData = {
            startTime: timelineStart.value,
            endTime: timelineEnd.value,
            tick: 12,
          }
        }
        // 初始化每天的值为0
        const fullData = {};
        for (let timestamp = timelineStart.value; timestamp <= new Date(year, month - 1, days).getTime(); timestamp += millisecondsPerDay) {
          const date = new Date(timestamp);
          const year = date.getFullYear();
          const month = date.getMonth() + 1
          const day = date.getDate()
          const key = `${year}-${month}-${day}`;
          fullData[key] = 0; // 初始化默认值
        }
        // 合并已有数据
        for (const key in rawData) {
          fullData[key] = rawData[key];
        }
        const data = Object.entries(fullData).map(([dateStr, value]) => {
          const timestamp = new Date(dateStr).getTime();
          return [timestamp, value];
        }).sort((a, b) => a[0] - b[0])
        // 转换为 [timestamp, value] 数组
        if (selected.value === 'month') {
          cacheData.value.monthData.data = data
        } else if (selected.value === 'quarter') {
          cacheData.value.quarterData.data = data
        } else if (selected.value === 'year') {
          cacheData.value.yearData.data = data
        }
        series.value = [{
          name: '异常站点个数',
          data: data
        }]
      })
      .catch(error => {
        console.log(error);
      });
}

function chageChart(selectValue) {
  selected.value = selectValue
  if (selectValue === 'month') {
    if (Object.keys(cacheData.value.monthData).length > 0) {
      chageChartOption(
          cacheData.value.monthData.startTime,
          cacheData.value.monthData.endTime,
          cacheData.value.monthData.tick
      )
      series.value = [{
        name: '异常站点个数',
        data: cacheData.value.monthData.data
      }]
      return
    }
  } else if (selectValue === 'quarter') {
    if (Object.keys(cacheData.value.quarterData).length > 0) {
      chageChartOption(
          cacheData.value.quarterData.startTime,
          cacheData.value.quarterData.endTime,
          cacheData.value.quarterData.tick
      )
      series.value = [{
        name: '异常站点个数',
        data: cacheData.value.quarterData.data
      }]
      return
    }
  } else if (selectValue === 'year') {
    if (Object.keys(cacheData.value.yearData).length > 0) {
      chageChartOption(
          cacheData.value.yearData.startTime,
          cacheData.value.yearData.endTime,
          cacheData.value.yearData.tick
      )
      series.value = [{
        name: '异常站点个数',
        data: cacheData.value.yearData.data
      }]
      return
    }
  }
  fetchdata()
}

onMounted(async () => {
  await fetchdata()
})
const chartOptions = ref(null)
</script>

<style scoped>
.area-chart {
  width: 100%;
}
</style>
