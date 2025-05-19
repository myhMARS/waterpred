<template>
  <admin-layout>
    <PageBreadcrumb :pageTitle="currentPageTitle"/>
    <div class="grid grid-cols-12 gap-4 md:gap-6">
      <div class="col-span-12">
        <AreaMetrics
            :time="data.endDate"
            :temperature="Number(data.temperature.at(-1)?.y)"
            :humidity="Number(data.humidity.at(-1)?.y)"
            :windpower="Number(data.windpower.at(-1)?.y)"
        />
      </div>
      <div class="col-span-12 space-y-6">
        <DatasChart :data="data"/>
      </div>
      <div class="col-span-12 space-y-6 xl:col-span-8">
        <ComponentCard title="相关站点">
          <RelatedSitesTable :county="String(county)"/>
        </ComponentCard>
      </div>

      <div class="col-span-12 space-y-6 xl:col-span-4">

        <WindChart :data="data"/>

      </div>
    </div>
  </admin-layout>
</template>

<script setup>
import {onMounted, ref} from "vue";
import {useRoute} from "vue-router";
import AdminLayout from '@/components/layout/AdminLayout.vue'
import RelatedSitesTable from '@/components/area/RelatedSitesTable.vue'
import PageBreadcrumb from "@/components/common/PageBreadcrumb.vue";
import DatasChart from "@/components/area/DatasChart.vue";
import AreaMetrics from "@/components/area/AreaMetrics.vue";
import WindChart from "@/components/area/WindChart.vue";
import axios from "axios";
import ComponentCard from "@/components/common/ComponentCard.vue";

const route = useRoute()
const county = route.params.county
console.log(county)
const currentPageTitle = ref('')
currentPageTitle.value = String(county)
const data = ref({
  startDate: '',
  endDate: '',
  temperature: [],
  humidity: [],
  windpower: [],
  winddirection: new Array(8).fill(0)
})

async function fetchData() {
  try {
    const response = await axios.get("/api/areadetail/", {
      params: {
        county: county
      }
    })
    data.value.startDate = new Date(response.data[0]?.times).toLocaleString()
    data.value.endDate = new Date(response.data.at(-1)?.times).toLocaleString()
    for (const record of response.data) {
      data.value.winddirection[Number(record.winddirection) - 1] += 1
      data.value.temperature.push({
        x: new Date(record.times).getTime(),
        y: record.temperature
      })
      data.value.humidity.push({
        x: new Date(record.times).getTime(),
        y: record.humidity
      })
      data.value.windpower.push({
        x: new Date(record.times).getTime(),
        y: record.windpower
      })
    }

  } catch (error) {
    console.log(error)
  }
}

onMounted(async () => {
  await fetchData()
})
</script>
