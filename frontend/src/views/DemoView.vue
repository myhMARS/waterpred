<template>
  <Header/>
  <section class="m-10 mt-16 grid grid-cols-1 md:grid-cols-2 gap-6">
    <Chart v-for="(chart, index) in charts" :key="index" :data="chart"/>
  </section>
</template>

<script>
import VueApexCharts from "vue3-apexcharts";
import Chart from "@/components/Chart.vue";
import Header from "@/components/Header.vue";
import axios from "axios";

const current_waterlevel = ["水位", "#4CAF50"]
const flood_limit_water_level = ["汛限水位", "#FA0303F9"]
const guaranteed_water_level = ["保证水位", "#FA0303F9"]
const warning_water_level = ["警戒水位", "#FAD300AF"]

class CustomChart {
  constructor(title, categories, ylabel, info) {
    this.title = title;
    this.categories = categories;
    this.ylabel = ylabel;
    this.Data = [];

    for (let i = 0; i < info.length; i++) {
      const data = {
        name: info[i].type[0],
        color: info[i].type[1],
        data: info[i].data
      };
      this.Data.push(data);
    }
  }
}

export default {
  components: {
    Header,
    apexchart: VueApexCharts,
    Chart,
  },
  data() {
    return {
      charts: [],
    };
  },
  mounted() {
    axios.get('/api/waterinfo/')
        .then(response => {
          const data = response.data;
          const times = [];
          const waterlevels = [];       // 桥东村水位
          const waterlevels63000100 = []; // 里畈水库当前水位
          const waterlevels63000120 = []; // 里畈东坑溪水位

          data.forEach(item => {
            times.push(new Date(item.times).toLocaleString('zh-CN'));
            waterlevels.push(item.waterlevels);
            waterlevels63000120.push(item.waterlevels63000120);
            waterlevels63000100.push(item.waterlevels63000100);
          });
          console.log(this.charts)
          // 更新图表数据
          const chart1 = new CustomChart(
              "里畈东坑溪水位",
              times,
              'mm',
              [{type: current_waterlevel, data: waterlevels63000120}]
          );
          // const chart2 = new chart("里畈水库水位",times,'mm',["水位"],[waterlevels], ["#4CAF50"])
          const chart2 = new CustomChart(
              "里畈水库水位",
              times,
              'mm',
              [
                {type: current_waterlevel, data: waterlevels63000100},
                {type: flood_limit_water_level,data: new Array(times.length).fill(234.73)}
              ]
          );
          const chart3 = new CustomChart(
              "桥东村水位",
              times,
              'mm',
              [
                {type:current_waterlevel, data: waterlevels},
                {type: guaranteed_water_level, data: new Array(times.length).fill(85.66)},
                {type:warning_water_level,data: new Array(times.length).fill(84.66)}
              ]
          )
          this.charts.push(chart1,chart2,chart3)
        })
        .catch(error => {
          console.error('请求失败:', error);
        });
  },
}
</script>

<style>
/* 可在这里添加额外的 CSS 样式 */
</style>
