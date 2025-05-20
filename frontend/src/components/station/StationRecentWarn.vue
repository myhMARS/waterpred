<template>
  <div
      class="overflow-hidden rounded-2xl border border-gray-200 bg-white px-4 pb-3 pt-4 dark:border-gray-800 dark:bg-white/[0.03] sm:px-6"
  >
    <div class="flex flex-col gap-2 mb-4 sm:flex-row sm:items-center sm:justify-between">
      <div>
        <h3 class="text-lg font-semibold text-gray-800 dark:text-white/90">近期告警</h3>
      </div>

      <div class="flex items-center gap-3">
        <button
            @click="toggleFilter"
            class="inline-flex items-center gap-2 rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-theme-sm font-medium text-gray-700 shadow-theme-xs hover:bg-gray-50 hover:text-gray-800 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-400 dark:hover:bg-white/[0.03] dark:hover:text-gray-200"
        >
          <svg
              class="stroke-current fill-white dark:fill-gray-800"
              width="20"
              height="20"
              viewBox="0 0 20 20"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
          >
            <path
                d="M2.29004 5.90393H17.7067"
                stroke=""
                stroke-width="1.5"
                stroke-linecap="round"
                stroke-linejoin="round"
            />
            <path
                d="M17.7075 14.0961H2.29085"
                stroke=""
                stroke-width="1.5"
                stroke-linecap="round"
                stroke-linejoin="round"
            />
            <path
                d="M12.0826 3.33331C13.5024 3.33331 14.6534 4.48431 14.6534 5.90414C14.6534 7.32398 13.5024 8.47498 12.0826 8.47498C10.6627 8.47498 9.51172 7.32398 9.51172 5.90415C9.51172 4.48432 10.6627 3.33331 12.0826 3.33331Z"
                fill=""
                stroke=""
                stroke-width="1.5"
            />
            <path
                d="M7.91745 11.525C6.49762 11.525 5.34662 12.676 5.34662 14.0959C5.34661 15.5157 6.49762 16.6667 7.91745 16.6667C9.33728 16.6667 10.4883 15.5157 10.4883 14.0959C10.4883 12.676 9.33728 11.525 7.91745 11.525Z"
                fill=""
                stroke=""
                stroke-width="1.5"
            />
          </svg>

          = 筛选
        </button>

        <div
            v-if="showFilter"
            class="fixed inset-0 z-50 flex items-center justify-center"
        >
          <div
              class="w-96 rounded-lg bg-white p-6 shadow-sm border border-gray-200 dark:bg-gray-900 dark:border-gray-800"
          >
            <form @submit.prevent="applyFilters">
              <div class="mb-4">
                <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">通知状态</label>
                <select
                    v-model="filters.isSuccess"
                    class="mt-1 block w-full rounded-md border-gray-300 border dark:bg-gray-800 dark:border-gray-600 dark:text-white"
                >
                  <option :value="''">全部</option>
                  <option :value="false">发送失败</option>
                  <option :value="true">已发送</option>
                </select>
              </div>
              <div class="mb-4">
                <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">告警状态</label>
                <select
                    v-model="filters.isCancel"
                    class="mt-1 block w-full rounded-md border-gray-300 border dark:bg-gray-800 dark:border-gray-600 dark:text-white"
                >
                  <option :value="''">全部</option>
                  <option :value="false">未确认</option>
                  <option :value="true">已确认</option>
                </select>
              </div>
              <div class="flex justify-end gap-2">
                <button
                    type="submit"
                    class="px-3 py-1 text-sm rounded-md bg-blue-600 text-white hover:bg-blue-700"
                >
                  确定
                </button>
                <button
                    type="button"
                    @click="toggleFilter"
                    class="px-3 py-1 text-sm rounded-md bg-gray-200 text-gray-800 hover:bg-gray-300 dark:bg-gray-700 dark:text-white dark:hover:bg-gray-600"
                >
                  取消
                </button>
              </div>
            </form>
          </div>
        </div>

        <button
            @click="clearFilters"
            class="inline-flex items-center gap-2 rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-theme-sm font-medium text-gray-700 shadow-theme-xs hover:bg-gray-50 hover:text-gray-800 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-400 dark:hover:bg-white/[0.03] dark:hover:text-gray-200"
        >
          查看全部
        </button>
      </div>
    </div>

    <div class="max-w-full overflow-x-auto custom-scrollbar">
      <table class="min-w-full">
        <thead>
        <tr class="border-t border-gray-100 dark:border-gray-800">
          <th class="py-3 text-left">
            <p class="font-medium text-gray-500 text-theme-xs dark:text-gray-400">通知状态</p>
          </th>
          <th class="py-3 text-left">
            <p class="font-medium text-gray-500 text-theme-xs dark:text-gray-400">最高水位</p>
          </th>
          <th class="py-3 text-left">
            <p class="font-medium text-gray-500 text-theme-xs dark:text-gray-400">告警状态</p>
          </th>
        </tr>
        </thead>
        <tbody>
        <tr
            v-for="(warning, index) of filter_warnings"
            :key="index"
            class="border-t border-gray-100 dark:border-gray-800"
        >
          <td class="py-3 whitespace-nowrap">
            <div>
              <span
                  :class="{
                  'rounded-full px-2 py-0.5 text-theme-xs font-medium': true,
                  'bg-success-50 text-success-600 dark:bg-success-500/15 dark:text-success-500':
                    warning.isSuccess === true,
                  'bg-error-50 text-error-600 dark:bg-error-500/15 dark:text-error-500':
                    warning.isSuccess === false,
                }"
              >
                {{ warning.isSuccess ? '已发送' : '发送失败' }}
              </span>
              <p
                  v-if="warning.isSuccess"
                  class="text-gray-500 text-theme-xs dark:text-gray-400">
                告警类型: {{ warning.noticetype }} | 通知时间: {{ warning.noticetime }}
              </p>
            </div>
          </td>
          <td class="py-3 whitespace-nowrap">
            <p class="text-gray-500 text-theme-sm dark:text-gray-400">{{ warning.max_level }}</p>
          </td>

          <td class="py-3 whitespace-nowrap">
            <div>
              <span
                  :class="{
                  'rounded-full px-2 py-0.5 text-theme-xs font-medium': true,
                  'bg-success-50 text-success-600 dark:bg-success-500/15 dark:text-success-500':
                    warning.isCancel === true,
                  'bg-error-50 text-error-600 dark:bg-error-500/15 dark:text-error-500':
                    warning.isCancel === false,
                }"
              >
                {{ warning.isCancel ? '已取消' : '告警中' }}
              </span>
              <p
                  v-if="warning.isCancel"
                  class="text-gray-500 text-theme-xs dark:text-gray-400">
                处理人: {{ warning.executor }} | 确认时间: {{ warning.canceltime }}
              </p>
            </div>
          </td>
        </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import {onMounted, ref} from 'vue'
import axios from "axios";

const showFilter = ref(false)

const {stationid} = defineProps({
  stationid: {
    type: String,
    required: true
  }
})

const filters = ref({
  station: stationid,
  isCancel: '',
  isSuccess: ''
})
console.log(filters.value)
const toggleFilter = () => {
  showFilter.value = !showFilter.value
}

const clearFilters = () => {
  filter_warnings.value = warnings.value
}

const applyFilters = () => {
  console.log('Applied filters:', filters.value)
  filter_warnings.value = filter_warnings.value.filter(warning => {
    const matchSuccess = filters.value.isSuccess === '' || warning.isSuccess === filters.value.isSuccess;
    const matchCancel = filters.value.isCancel === '' || warning.isCancel === filters.value.isCancel;

    return matchSuccess && matchCancel;
  });
  showFilter.value = false
}

const warnings = ref([])
const filter_warnings = ref([])

async function fetchData() {
  try {
    const response = await axios.get(
        '/api/warnings/', {
          params: filters.value,
          headers: {
            Authorization: `JWT ${localStorage.getItem('token')}`
          }
        })
    console.log(response.data)
    const warnings_data = response.data
    for (const warning of warnings_data) {
      warnings.value.push({
        name: warning.station_name,
        id: warning.station,
        max_level: warning.max_level,
        isSuccess: warning.isSuccess,
        noticetype: warning.noticetype,
        noticetime: new Date(warning.noticetime).toLocaleString(),
        isCancel: warning.isCanceled,
        executor: warning.executor,
        canceltime: new Date(warning.canceltime).toLocaleString(),
      })
    }
    filter_warnings.value = warnings.value
  } catch (error) {
    console.log(error)
  }
}

onMounted(async () => {
  await fetchData()
})
</script>
