<template>
  <admin-layout>
    <div class="grid grid-cols-12 gap-4 md:gap-6">
      <div class="col-span-12 h-full xl:col-span-3">
        <StationMetric/>
      </div>

      <div class="col-span-12 xl:col-span-9">
        <statistics-chart/>
      </div>

      <div class="col-span-12">
        <recent-warn/>
      </div>
    </div>
  </admin-layout>
</template>

<script setup>
import AdminLayout from '@/components/layout/AdminLayout.vue'
import StatisticsChart from '@/components/dashboard/StatisticsChart.vue'
import RecentWarn from '@/components/dashboard/RecentWarn.vue'
import StationMetric from "@/components/dashboard/StationMetric.vue";

defineOptions({
  name: 'Dashborad'
})
</script>
