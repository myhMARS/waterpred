<template>
  <Header />
  <section class="m-10 mt-16 grid grid-cols-1 md:grid-cols-2 gap-6">
    <Chart v-for="(chart, index) in charts" :key="index" :data="chart" />
  </section>
</template>

<script>
import VueApexCharts from "vue3-apexcharts";
import Chart from "../components/Chart.vue";
import Header from "@/components/Header.vue";

export default {
  components: {
    Header,
    apexchart: VueApexCharts,
    Chart,
  },
  data() {
    return {
      charts: [
        {
          title: "里畈东坑溪水位",
          categories: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
          ylabel: "mm",
          Data: [
            {
              name: "水位",
              data: [236.07, 236.06, 236.06, 236.06, 236.06, 236.06, 236.05, 236.05, 236.05, 236.06, 236.06, 236.21]
,
            },
          ],
        },
        {
          title: "里畈水库水位",
          ylabel: "mm",
          categories: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
          Data: [
            {
              name: "汛限水位",
              color: '#FA0303F9',
              data: new Array(12).fill(234.73),
            },
            {
              name: "当前水位",
              data: [199.66, 199.66, 199.65, 199.65, 199.65, 199.64, 199.64, 199.63, 199.62, 199.61, 199.6, 199.59],
            },
          ],
        },
        {
          title: "桥东村水位",
          ylabel: "mm",
          categories: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
          Data: [
            {
              name: "保证水位",
              color: "#FA0303F9",
              data: new Array(12).fill(85.66),
            },
            {
              name: "警戒水位",
              color: "#FAD300AF",
              data: new Array(12).fill(84.66),
            },
            {
              name: "水位",
              color: "#01cffb",
              data: [83.39, 83.41, 83.42, 83.43, 83.44, 83.45, 83.46, 83.47, 83.48, 83.49, 83.49, 83.49]

            },
          ],
        },
      ],
    };
  },
};
</script>

<style>
/* 可在这里添加额外的 CSS 样式 */
</style>
