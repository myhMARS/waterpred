<template>
  <AdminLayout>
    <PageBreadcrumb :pageTitle="currentPageTitle"/>

    <div class="space-y-5 sm:space-y-6">
      <div class="overflow-hidden">
        <div
            class="flex flex-col gap-2 px-4 py-4 border border-b-0 border-gray-200 rounded-b-none rounded-xl dark:border-gray-800 sm:flex-row sm:items-center sm:justify-between"
            id="container"></div>
      </div>
    </div>
  </AdminLayout>
</template>

<script setup>
import {onMounted, ref} from 'vue'
import AdminLayout from "@/components/layout/AdminLayout.vue";
import PageBreadcrumb from "@/components/common/PageBreadcrumb.vue";
import AMapLoader from '@amap/amap-jsapi-loader'
import axios from "axios";

const currentPageTitle = ref('区域地图')
window._AMapSecurityConfig = {
  securityJsCode: import.meta.env.VITE_APP_AMAP_SECURITY_CODE,
};

const amapApiKey = import.meta.env.VITE_APP_AMAP_API_KEY;

onMounted(async () => {
  const AMap = await AMapLoader.load({
    key: amapApiKey,
    version: '2.0',
    plugins: ['AMap.GeoJSON']
  })

  const map = new AMap.Map('container', {
    zoom: 6,
    center: [120.2, 30.3],
    viewMode: '2D'
  })

  try {
    // ✅ 加载本地或网络 GeoJSON 文件
    const res = await axios.get('/geo/zhejiang.json')  // 改成你的路径
    const geojson = res.data

    const geoJsonLayer = new AMap.GeoJSON({
      geoJSON: geojson,
      getPolygon: (geojson, lnglats) => {
        return new AMap.Polygon({
          path: lnglats,
          strokeColor: '#1791fc',
          fillColor: '#1791fc',
          fillOpacity: 0.05,
          strokeWeight: 1
        })
      }
    })

    map.add(geoJsonLayer)
  } catch (err) {
    console.error('GeoJSON 加载失败:', err)
  }
})
</script>

<style scoped>
#container {
  width: 100%;
  height: 630px;
}

</style>
