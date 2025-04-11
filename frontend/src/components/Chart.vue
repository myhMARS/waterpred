<template>
  <div class="relative flex flex-col bg-white rounded-xl shadow-md text-gray-700">
    <div class="p-6">
      <h3 class="text-xl font-semibold text-blue-gray-900 mb-4">{{ data.title }}</h3>
      <apexchart width="100%" height="350" type="area" :options="chartOptions" :series="series"></apexchart>
    </div>
  </div>
</template>

<script>
import VueApexCharts from "vue3-apexcharts";

export default {
  name: 'AreaChartCard',
  components: {
    apexchart: VueApexCharts,
  },
  props: {
    data: {
      type: Object,
      required: true,
      validator: (value) => {
        return value.title && value.Data && value.categories
      }
    }
  },
  computed: {
    chartOptions() {
      const options = {
        colors: ["#4CAF50", "#2196F3"],
        chart: {
          height: 350,
          type: "area",
          zoom: {enabled: false},
          toolbar: {show: false},
          animations: {
            enabled: true,
            easing: 'easeinout',
            speed: 800
          }
        },
        dataLabels: {enabled: false},
        legend: {
          show: true,
          showForSingleSeries: true,
          position: 'top',
          horizontalAlign: 'right',
          fontSize: '10px',
          markers: {
            radius: 12
          }
        },
        stroke: {
          curve: "smooth",
          width: 2,
          lineCap: 'round'
        },
        grid: {
          show: true,
          borderColor: "#EEEEEE",
          strokeDashArray: 5,
          padding: {top: 20, right: 20},
          yaxis: {
            lines: {
              show: true
            }
          }
        },
        tooltip: {
          theme: "light",
          shared: true,
          intersect: false,
          y: {
            formatter: (value) => {
              return value.toLocaleString()
            }
          }
        },
        yaxis: {
          labels: {
            style: {
              colors: "#757575",
              fontSize: "12px",
              fontWeight: 300
            },
            formatter: (value) => {
              return value.toLocaleString()
            }
          },
          title: {
            text: this.data.ylabel
          }
        },
        xaxis: {
          categories: this.data.categories,
          labels: {
            style: {
              colors: "#757575",
              fontSize: "12px",
              fontWeight: 300
            }
          },
          axisBorder: {
            show: false
          },
          axisTicks: {
            show: false
          }
        },
        fill: {
          type: "gradient",
          gradient: {
            shadeIntensity: 1,
            opacityFrom: 0.7,
            opacityTo: 0.3,
            stops: [0, 0, 100]
          },
        },
        annotations: {
          xaxis: []
        }
      };

      if (this.data.predline) {
        options.annotations.xaxis.push({
          x: this.data.predline,
          borderColor: 'rgba(128, 128, 128, 0.5)',
          label: {
            orientation: 'horizontal',
            style: {
              color: '#fff',
              background: 'rgba(128, 128, 128, 0.5)',
            },
            text: '预测起点',
          },
        });
      }

      return options;
    },
    series() {
      return this.data.Data || [];
    }
  }

};
</script>

<style scoped>
/* 添加卡片悬停效果 */
.relative:hover {
  transform: translateY(-2px);
  transition: transform 0.2s ease;
  box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
}
</style>
