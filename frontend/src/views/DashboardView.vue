<template>
  <admin-layout>
    <div class="grid grid-cols-12 gap-4 md:gap-6">
      <div class="col-span-12 space-y-6 xl:col-span-12">
        <StationMetric/>
      </div>

      <div class="col-span-12">
        <statistics-chart/>
      </div>

      <div class="col-span-12 xl:col-span-5">
        <customer-demographic/>
      </div>

      <div class="col-span-12 xl:col-span-7">
        <recent-orders/>
      </div>
    </div>
  </admin-layout>
</template>

<script setup>
import AdminLayout from '@/components/layout/AdminLayout.vue'
import CustomerDemographic from '@/components/waterinfos/CustomerDemographic.vue'
import StatisticsChart from '@/components/waterinfos/StatisticsChart.vue'
import RecentOrders from '@/components/waterinfos/RecentOrders.vue'
import StationMetric from "@/components/waterinfos/StationMetric.vue";

defineOptions({
  name: 'Dashborad'
})
</script>
