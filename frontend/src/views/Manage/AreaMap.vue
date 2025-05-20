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
import {useRouter} from 'vue-router'

const router = useRouter()

const currentPageTitle = ref('区域地图')
window._AMapSecurityConfig = {
  securityJsCode: import.meta.env.VITE_APP_AMAP_SECURITY_CODE,
};

const amapApiKey = import.meta.env.VITE_APP_AMAP_API_KEY;
const map = ref(null)
const currentGeoJsonLayer = ref(null)

function clearGeoJsonLayer() {
  if (currentGeoJsonLayer.value) {
    map.value.remove(currentGeoJsonLayer.value)
    currentGeoJsonLayer.value = null
  }
}

async function loadAMap() {
  try {
    return await AMapLoader.load({
      key: amapApiKey,
      version: "2.0",
      plugins: [
        "AMap.DistrictSearch", "AMap.MapType", "AMap.LngLat",
        "AMap.LabelsLayer", "AMap.Geocoder", "AMap.Util", "AMap.CanvasLayer", "AMap.GeoJSON"
      ],
    })
  } catch (error) {
    console.log(error)
  }
}

async function getStationCount(location) {
  let params = {}
  if (location.slice(-1) === '市') {
    params = {
      city: location.slice(0, -1)
    }
  } else {
    params = {
      county: location.slice(0, -1)
    }
  }

  try {
    const response = await axios.get("/api/areastationcount/", {
      params: params,
      headers: {
        Authorization: `JWT ${localStorage.getItem('token')}`
      }
    })
    return response.data
  } catch (error) {
    console.error(error)
    return 0
  }
}

async function initMap(AMap) {
  try {
    const district = new AMap.DistrictSearch({subdistrict: 1, extensions: 'all', level: 'province'});
    return new Promise((resolve, reject) => {
      district.search('浙江省', function (status, result) {
        const bounds = result.districtList[0].boundaries
        const mask = []
        for (let i = 0; i < bounds.length; i++) {
          mask.push([bounds[i]])
        }
        const map = new AMap.Map("container", {  // 设置地图容器id
          mask: mask, // 为Map实例制定掩模的路径,各图层将值显示路径范围内图像,3D模式下有效
          zoom: 12, // 设置当前显示级别
          expandZoomRange: true, // 开启显示范围设置
          zooms: [8, 20], //最小显示级别为7，最大显示级别为20
          viewMode: "3D",    // 特别注意,设置为3D则其他地区不显示
          zoomEnable: true, // 是否可以缩放地图
          resizeEnable: true,
          showLabel: false
        });
        map.setCity('浙江省')
        map.addControl(new AMap.MapType({
          defaultType: 1 //0代表默认，1代表卫星
        }));
        map.setMapStyle('amap://styles/normal')
        map.on('zoomend', async () => {
          const zoom = map.getZoom()
          await loadGeoJsonByZoom(AMap, map, zoom)
        })
        // 添加描边
        for (let i = 0; i < bounds.length; i++) {
          const polyline = new AMap.Polyline({
            path: bounds[i],
            strokeColor: '#3078AC',
            strokeWeight: 2,
          })
          polyline.setMap(map);
        }
        resolve(map)
      })

    })

  } catch (error) {
    console.log(error)
  }
}

async function loadGeoJsonByZoom(AMap, map, zoom) {
  let geojsonUrl = ''

  if (zoom <= 9) {
    geojsonUrl = '/geo/zhejiang-city.json'      // 省级
  } else {
    geojsonUrl = '/geo/zhejiang.json'      // 市级或县级
  }

  const res = await axios.get(geojsonUrl)
  const geojson = res.data

  clearGeoJsonLayer()

  const geoJsonLayer = new AMap.GeoJSON({
    geoJSON: geojson,
    getPolygon: (feature, lnglats) => {
      return new AMap.Polygon({
        path: lnglats,
        strokeColor: '#1791fc',
        fillColor: '#1791fc',
        fillOpacity: 0.05,
        strokeWeight: 1,
        extData: feature
      })
    }
  })

  map.add(geoJsonLayer)
  currentGeoJsonLayer.value = geoJsonLayer

  geoJsonLayer.eachOverlay(async (polygon) => {
    const name = polygon.getExtData()?.properties?.name || '未知'
    const stationcount = await getStationCount(name)
    polygon.setOptions({
      fillOpacity: stationcount.count / 10 + 0.05,
      fillColor: stationcount.station_status === 0 ? '#1791fc' : '#ef4444'
    })
    const label = new AMap.Text({
      text: name + ',站点数量:' + String(stationcount.count) + '/异常数量:' + String(stationcount.station_status),
      position: polygon.getBounds().getCenter(),
      style: {
        background: 'rgba(255,255,255,0.8)',
        padding: '2px 6px',
        fontSize: '14px',
        color: '#333',
        borderRadius: '4px',
      },
      zIndex: 1100
    })

    polygon.on('mouseover', () => {
      polygon.setOptions({
        fillColor: stationcount.station_status === 0 ? '#1791fc' : '#ef4444',
        fillOpacity: stationcount.station_status / 100 + 0.05 + 0.4,
        strokeWeight: 2,
        strokeColor: '#ff6600'
      })
      label.setMap(map)
    }, {passive: true})
    if (name.slice(-1) !== '市') {
      polygon.on('click', () => {
        router.push(`/area/${name.slice(0, -1)}`)
      })
    }

    polygon.on('mouseout', () => {
      polygon.setOptions({
        fillOpacity: stationcount.count / 10 + 0.05,
        fillColor: stationcount.station_status === 0 ? '#1791fc' : '#ef4444',
        strokeWeight: 1,
        strokeColor: '#1791fc'
      })
      label.setMap(null)
    }, {passive: true})
  })
}


onMounted(async () => {
  const AMap = await loadAMap()
  map.value = await initMap(AMap)
  await loadGeoJsonByZoom(AMap, map.value, map.value.getZoom())
})
</script>

<style scoped>
#container {
  width: 100%;
  height: 630px;
}

</style>
