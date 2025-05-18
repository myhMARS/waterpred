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
import {onMounted, onUnmounted, ref} from "vue";
import AdminLayout from "@/components/layout/AdminLayout.vue";
import PageBreadcrumb from "@/components/common/PageBreadcrumb.vue";
import AMapLoader from "@amap/amap-jsapi-loader";
import axios from "axios";
import {useRouter} from "vue-router";

const currentPageTitle = ref('站点地图')
const router = useRouter()

window._AMapSecurityConfig = {
  securityJsCode: import.meta.env.VITE_APP_AMAP_SECURITY_CODE,
};
const amapApiKey = import.meta.env.VITE_APP_AMAP_API_KEY;
let map = null;

async function getStations() {
  try {
    const response = await axios.get('/api/stationlist/')
    let stations = []
    for (const station of response.data) {
      stations.push({
        id: station.id,
        name: station.name,
        status: station.status
      })
    }
    return stations
  } catch (error) {
    console.log(error)
  }
}

async function loadAMap() {
  try {
    return await AMapLoader.load({
      key: amapApiKey,
      version: "2.0",
      plugins: ["AMap.Scale", "AMap.DistrictSearch", "AMap.MapType", "AMap.LngLat", "AMap.LabelsLayer", "AMap.Geocoder"],
    })
  } catch (error) {
    console.log(error)
  }
}

async function getLocation(AMap, stations) {
  try {
    for (let station of stations) {
      const response = await axios.get('/api/getlocation/', {
        params: {
          station_name: station.name
        },
      })
      station.position = [response.data.qt[0].jd, response.data.qt[0].wd]
    }
    return stations
  } catch (error) {
    console.log(error)
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

function setMarkers(AMap, map, stations) {
  const labelsLayer = new AMap.LabelsLayer({
    zooms: [3, 20],
    zIndex: 1000,
    collision: false, //该层内标注是否避让
    allowCollision: true, //不同标注层之间是否避让
  });
  let labelMarkers = []
  for (const station of stations) {
    console.log('station', station)
    const icon = {
      type: "image", //图标类型，现阶段只支持 image 类型
      image: "/images/icons/station.png", //可访问的图片 URL
      size: [30, 30], //图片尺寸
      anchor: "center", //图片相对 position 的锚点，默认为 bottom-center
    };
    const text = {
      content: station.name, //要展示的文字内容
      direction: "right", //文字方向，有 icon 时为围绕文字的方向，没有 icon 时，则为相对 position 的位置
      style: {
        fontSize: 12, //字体大小
        fillColor: station.status===0?"#22886f":"#ef4444",
        strokeColor: "#fff", //描边颜色
        strokeWidth: 2, //描边宽度
      },
    };
    const labelMarker = new AMap.LabelMarker({
      name: station.id, //此属性非绘制文字内容，仅为标识使用
      position: station.position,
      zIndex: 16,
      rank: 1, //避让优先级
      icon: icon, //标注图标，将 icon 对象传给 icon 属性
      text: text, //标注文本，将 text 对象传给 text 属性
    });
    labelMarker.on('click', function (e) {
      router.push('/station/'+station.id)
    });
    labelMarkers.push(labelMarker)
    labelsLayer.add(labelMarkers);
    map.add(labelsLayer);
  }
}

onMounted(async () => {
  let stations = await getStations()
  const AMap = await loadAMap()
  map = await initMap(AMap)
  stations = await getLocation(AMap, stations)
  setMarkers(AMap, map, stations)
});

onUnmounted(() => {
  if (map) {
    map.destroy(); // 清理地图实例，避免内存泄漏
  }
});

</script>

<style scoped>
#container {
  width: 100%;
  height: 630px;
}

</style>
