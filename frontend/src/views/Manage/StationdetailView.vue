<template>
  <admin-layout>
    <PageBreadcrumb :pageTitle="currentPageTitle"/>
    <div class="space-y-5 sm:space-y-6">
      <StationMetrics :data="metrics"/>
      <div class="gap-6 space-y-5 sm:space-y-6 xl:grid xl:grid-cols-12 xl:space-y-0">
        <div class="xl:col-span-7 2xl:col-span-8">
          <div class="space-y-5  sm:space-y-6">
            <div class="grid sm:gap-6 lg:grid-cols-2">
              <RainRate :stationid="String(stationid)"/>
              <WaterlevelRate :stationid="String(stationid)"/>
            </div>

            <!-- {/* Funnel */} -->
            <WaterLevelChart :series="waterlevels"/>
            <!-- {/* Table */} -->
            <RecentWarn :stationid="String(stationid)"/>
          </div>
        </div>
        <div class="space-y-5 sm:space-y-6 xl:col-span-5 2xl:col-span-4">
          <PerformanceTab :rainData="rains" :predData="pred" :status="status" :station_id="String(stationid)"/>
        </div>
      </div>
    </div>
  </admin-layout>
</template>

<script setup>
import AdminLayout from '@/components/layout/AdminLayout.vue'
import StationMetrics from '@/components/station/StationMetrics.vue'
import RainRate from '@/components/station/RainRate.vue'
import WaterlevelRate from '@/components/station/WaterlevelRate.vue'
import RecentWarn from '@/components/station/StationRecentWarn.vue'
import WaterLevelChart from '@/components/station/WaterLevelChart.vue'
import PerformanceTab from '@/components/station/PerformanceTab.vue'
import PageBreadcrumb from "@/components/common/PageBreadcrumb.vue";
import {onMounted, ref} from "vue";
import {useRoute} from "vue-router";
import axios from "axios";

const route = useRoute()
const stationid = route.params.id
const currentPageTitle = ref('')
const metrics = ref({})
const status = ref(0)

async function fetchMetrics() {
  try {
    const response = await axios.get('/api/stationlist/', {
      params: {
        'stationid': stationid
      },
      headers: {
        Authorization: `JWT ${localStorage.getItem('token')}`
      }
    })
    currentPageTitle.value = response.data[0].name
    metrics.value = response.data[0]
    metrics.value.time = new Date(metrics.value.time).toLocaleString()
    status.value = response.data[0].status
  } catch (error) {
    console.log(error)
  }
}

const waterlevels = ref([
  {
    name: '水位',
    data: []
  },
  {
    name: '汛限水位',
    data: []
  },
  {
    name: '警戒水位',
    data: []
  },
  {
    name: '保证水位',
    data: []
  }
])

const rains = ref([{
  name: '降水量',
  data: []
}])
const pred = ref([{
  name: '水位',
  data: []
}])

async function fetchWaterLevelChart() {
  try {
    const response = await axios.get('/api/waterinfo/', {
      params: {
        'station_id': stationid,
        'length': 600
      },
      headers: {
        Authorization: `JWT ${localStorage.getItem('token')}`
      }
    })
    for (const record of response.data.data) {
      const waterlevelData = {
        x: new Date(record.times).getTime(),
        y: record.waterlevels
      }
      const rainData = {
        x: new Date(record.times).getTime(),
        y: record.rains
      }
      waterlevels.value[0].data.push(waterlevelData)
      if (metrics.value.flood_limit !== null) {
        waterlevels.value[1].data.push({
          x: new Date(record.times).getTime(),
          y: metrics.value.flood_limit
        })
      }
      if (metrics.value.warning !== null) {
        waterlevels.value[2].data.push({
          x: new Date(record.times).getTime(),
          y: metrics.value.warning
        })
      }

      if (metrics.value.guaranteed !== null) {
        waterlevels.value[3].data.push({
          x: new Date(record.times).getTime(),
          y: metrics.value.guaranteed
        })
      }

      rains.value[0].data.push(rainData)
    }

    const pred_starttime = new Date(response.data.data.at(-1).times).getTime()
    let timesplit = 3600000
    for (const record of response.data.pred) {
      const predData = {
        x: pred_starttime + timesplit,
        y: record.toFixed(2)
      }
      pred.value[0].data.push(predData)
      timesplit += 3600000
    }
  } catch (error) {
    console.log(error)
  }
}

onMounted(async () => {
  await fetchMetrics()
  await fetchWaterLevelChart()
})
</script>
