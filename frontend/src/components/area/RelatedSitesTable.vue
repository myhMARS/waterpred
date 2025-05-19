<template>
  <div
      class="overflow-hidden rounded-2xl border border-gray-200 bg-white px-5 pb-3 pt-4 dark:border-gray-800 dark:bg-white/[0.03] sm:px-6"
  >
    <div
        class="flex flex-col gap-2 px-4 py-4 border border-b-0 border-gray-200 rounded-b-none rounded-xl dark:border-gray-800 sm:flex-row sm:items-center sm:justify-between"
    >
      <div class="flex items-center gap-3">
        <span class="text-gray-500 dark:text-gray-400">展示</span>
        <div class="relative z-20 bg-transparent">
          <select
              v-model="perPage"
              class="w-full py-2 pl-3 pr-8 text-sm text-gray-800 bg-transparent border border-gray-300 rounded-lg appearance-none dark:bg-dark-900 h-9 bg-none shadow-theme-xs placeholder:text-gray-400 focus:border-brand-300 focus:outline-hidden focus:ring-3 focus:ring-brand-500/10 dark:border-gray-700 dark:bg-gray-900 dark:text-white/90 dark:placeholder:text-white/30 dark:focus:border-brand-800"
              :class="{ 'text-gray-500 dark:text-gray-400': perPage !== '' }"
          >
            <option value="10" class="text-gray-500 dark:bg-gray-900 dark:text-gray-400">10</option>
            <option value="8" class="text-gray-500 dark:bg-gray-900 dark:text-gray-400">8</option>
            <option value="5" class="text-gray-500 dark:bg-gray-900 dark:text-gray-400">5</option>
          </select>
          <span
              class="absolute z-30 text-gray-500 -translate-y-1/2 pointer-events-none right-2 top-1/2 dark:text-gray-400"
          >
            <svg
                class="stroke-current"
                width="16"
                height="16"
                viewBox="0 0 16 16"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
            >
              <path
                  d="M3.8335 5.9165L8.00016 10.0832L12.1668 5.9165"
                  stroke=""
                  stroke-width="1.2"
                  stroke-linecap="round"
                  stroke-linejoin="round"
              />
            </svg>
          </span>
        </div>
        <span class="text-gray-500 dark:text-gray-400">条数据</span>
      </div>

      <div class="flex flex-col gap-3 sm:flex-row sm:items-center">
        <div class="relative">
          <button class="absolute text-gray-500 -translate-y-1/2 left-4 top-1/2 dark:text-gray-400">
            <svg
                class="fill-current"
                width="20"
                height="20"
                viewBox="0 0 20 20"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
            >
              <path
                  fill-rule="evenodd"
                  clip-rule="evenodd"
                  d="M3.04199 9.37363C3.04199 5.87693 5.87735 3.04199 9.37533 3.04199C12.8733 3.04199 15.7087 5.87693 15.7087 9.37363C15.7087 12.8703 12.8733 15.7053 9.37533 15.7053C5.87735 15.7053 3.04199 12.8703 3.04199 9.37363ZM9.37533 1.54199C5.04926 1.54199 1.54199 5.04817 1.54199 9.37363C1.54199 13.6991 5.04926 17.2053 9.37533 17.2053C11.2676 17.2053 13.0032 16.5344 14.3572 15.4176L17.1773 18.238C17.4702 18.5309 17.945 18.5309 18.2379 18.238C18.5308 17.9451 18.5309 17.4703 18.238 17.1773L15.4182 14.3573C16.5367 13.0033 17.2087 11.2669 17.2087 9.37363C17.2087 5.04817 13.7014 1.54199 9.37533 1.54199Z"
                  fill=""
              />
            </svg>
          </button>
          <input
              v-model="search"
              type="text"
              placeholder="Search..."
              class="dark:bg-dark-900 h-11 w-full rounded-lg border border-gray-300 bg-transparent py-2.5 pl-11 pr-4 text-sm text-gray-800 shadow-theme-xs placeholder:text-gray-400 focus:border-brand-300 focus:outline-hidden focus:ring-3 focus:ring-brand-500/10 dark:border-gray-700 dark:bg-gray-900 dark:text-white/90 dark:placeholder:text-white/30 dark:focus:border-brand-800 xl:w-[300px]"
          />
        </div>
      </div>
    </div>

    <div class="max-w-full overflow-x-auto">
      <table class="w-full min-w-full">
        <thead>
        <tr>
          <!--- 站点编号 --->
          <th class="px-4 py-3 text-left border border-gray-100 dark:border-gray-800">
            <div
                class="flex items-center justify-between w-full cursor-pointer"
                @click="sortBy('id')"
            >
              <div class="flex items-center gap-3">
                <p class="font-medium text-gray-700 text-theme-xs dark:text-gray-400">站点编号</p>
              </div>
              <span class="flex flex-col gap-0.5">
                  <svg
                      class="fill-gray-300 dark:fill-gray-700"
                      width="8"
                      height="5"
                      viewBox="0 0 8 5"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                        d="M4.40962 0.585167C4.21057 0.300808 3.78943 0.300807 3.59038 0.585166L1.05071 4.21327C0.81874 4.54466 1.05582 5 1.46033 5H6.53967C6.94418 5 7.18126 4.54466 6.94929 4.21327L4.40962 0.585167Z"
                        fill=""
                    />
                  </svg>
                  <svg
                      class="fill-gray-300 dark:fill-gray-700"
                      width="8"
                      height="5"
                      viewBox="0 0 8 5"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                        d="M4.40962 4.41483C4.21057 4.69919 3.78943 4.69919 3.59038 4.41483L1.05071 0.786732C0.81874 0.455343 1.05582 0 1.46033 0H6.53967C6.94418 0 7.18126 0.455342 6.94929 0.786731L4.40962 4.41483Z"
                        fill=""
                    />
                  </svg>
                </span>
            </div>
          </th>
          <!--- 站点名称 --->
          <th class="px-4 py-3 text-left border border-gray-100 dark:border-gray-800">
            <div
                class="flex items-center justify-between w-full cursor-pointer"
                @click="sortBy('name')"
            >
              <div class="flex items-center gap-3">
                <p class="font-medium text-gray-700 text-theme-xs dark:text-gray-400">站点名称</p>
              </div>
              <span class="flex flex-col gap-0.5">
                  <svg
                      class="fill-gray-300 dark:fill-gray-700"
                      width="8"
                      height="5"
                      viewBox="0 0 8 5"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                        d="M4.40962 0.585167C4.21057 0.300808 3.78943 0.300807 3.59038 0.585166L1.05071 4.21327C0.81874 4.54466 1.05582 5 1.46033 5H6.53967C6.94418 5 7.18126 4.54466 6.94929 4.21327L4.40962 0.585167Z"
                        fill=""
                    />
                  </svg>
                  <svg
                      class="fill-gray-300 dark:fill-gray-700"
                      width="8"
                      height="5"
                      viewBox="0 0 8 5"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                        d="M4.40962 4.41483C4.21057 4.69919 3.78943 4.69919 3.59038 4.41483L1.05071 0.786732C0.81874 0.455343 1.05582 0 1.46033 0H6.53967C6.94418 0 7.18126 0.455342 6.94929 0.786731L4.40962 4.41483Z"
                        fill=""
                    />
                  </svg>
                </span>
            </div>
          </th>
          <!--- 站点位置 --->
          <!--- 汛限水位 --->
          <th class="px-4 py-3 text-left border border-gray-100 dark:border-gray-800">
            <div
                class="flex items-center justify-between w-full cursor-pointer"
                @click="sortBy('flood_limit')"
            >
              <p class="font-medium text-gray-700 text-theme-xs dark:text-gray-400">汛限水位</p>
              <span class="flex flex-col gap-0.5">
                  <svg
                      class="fill-gray-300 dark:fill-gray-700"
                      width="8"
                      height="5"
                      viewBox="0 0 8 5"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                        d="M4.40962 0.585167C4.21057 0.300808 3.78943 0.300807 3.59038 0.585166L1.05071 4.21327C0.81874 4.54466 1.05582 5 1.46033 5H6.53967C6.94418 5 7.18126 4.54466 6.94929 4.21327L4.40962 0.585167Z"
                        fill=""
                    />
                  </svg>
                  <svg
                      class="fill-gray-300 dark:fill-gray-700"
                      width="8"
                      height="5"
                      viewBox="0 0 8 5"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                        d="M4.40962 4.41483C4.21057 4.69919 3.78943 4.69919 3.59038 4.41483L1.05071 0.786732C0.81874 0.455343 1.05582 0 1.46033 0H6.53967C6.94418 0 7.18126 0.455342 6.94929 0.786731L4.40962 4.41483Z"
                        fill=""
                    />
                  </svg>
                </span>
            </div>
          </th>
          <!--- 警戒水位 --->
          <th class="px-4 py-3 text-left border border-gray-100 dark:border-gray-800">
            <div
                class="flex items-center justify-between w-full cursor-pointer"
                @click="sortBy('warning')"
            >
              <p class="font-medium text-gray-700 text-theme-xs dark:text-gray-400">警戒水位</p>
              <span class="flex flex-col gap-0.5">
                  <svg
                      class="fill-gray-300 dark:fill-gray-700"
                      width="8"
                      height="5"
                      viewBox="0 0 8 5"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                        d="M4.40962 0.585167C4.21057 0.300808 3.78943 0.300807 3.59038 0.585166L1.05071 4.21327C0.81874 4.54466 1.05582 5 1.46033 5H6.53967C6.94418 5 7.18126 4.54466 6.94929 4.21327L4.40962 0.585167Z"
                        fill=""
                    />
                  </svg>
                  <svg
                      class="fill-gray-300 dark:fill-gray-700"
                      width="8"
                      height="5"
                      viewBox="0 0 8 5"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                        d="M4.40962 4.41483C4.21057 4.69919 3.78943 4.69919 3.59038 4.41483L1.05071 0.786732C0.81874 0.455343 1.05582 0 1.46033 0H6.53967C6.94418 0 7.18126 0.455342 6.94929 0.786731L4.40962 4.41483Z"
                        fill=""
                    />
                  </svg>
                </span>
            </div>
          </th>
          <!--- 保证水位 --->
          <th class="px-4 py-3 text-left border border-gray-100 dark:border-gray-800">
            <div
                class="flex items-center justify-between w-full cursor-pointer"
                @click="sortBy('guaranteed')"
            >
              <p class="font-medium text-gray-700 text-theme-xs dark:text-gray-400">保证水位</p>
              <span class="flex flex-col gap-0.5">
                  <svg
                      class="fill-gray-300 dark:fill-gray-700"
                      width="8"
                      height="5"
                      viewBox="0 0 8 5"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                        d="M4.40962 0.585167C4.21057 0.300808 3.78943 0.300807 3.59038 0.585166L1.05071 4.21327C0.81874 4.54466 1.05582 5 1.46033 5H6.53967C6.94418 5 7.18126 4.54466 6.94929 4.21327L4.40962 0.585167Z"
                        fill=""
                    />
                  </svg>
                  <svg
                      class="fill-gray-300 dark:fill-gray-700"
                      width="8"
                      height="5"
                      viewBox="0 0 8 5"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                        d="M4.40962 4.41483C4.21057 4.69919 3.78943 4.69919 3.59038 4.41483L1.05071 0.786732C0.81874 0.455343 1.05582 0 1.46033 0H6.53967C6.94418 0 7.18126 0.455342 6.94929 0.786731L4.40962 4.41483Z"
                        fill=""
                    />
                  </svg>
                </span>
            </div>
          </th>
          <!--- 上报时间 --->
          <th class="px-4 py-3 text-left border border-gray-100 dark:border-gray-800">
            <div
                class="flex items-center justify-between w-full cursor-pointer"
                @click="sortBy('time')"
            >
              <p class="font-medium text-gray-700 text-theme-xs dark:text-gray-400">上报时间</p>
              <span class="flex flex-col gap-0.5">
                  <svg
                      class="fill-gray-300 dark:fill-gray-700"
                      width="8"
                      height="5"
                      viewBox="0 0 8 5"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                        d="M4.40962 0.585167C4.21057 0.300808 3.78943 0.300807 3.59038 0.585166L1.05071 4.21327C0.81874 4.54466 1.05582 5 1.46033 5H6.53967C6.94418 5 7.18126 4.54466 6.94929 4.21327L4.40962 0.585167Z"
                        fill=""
                    />
                  </svg>
                  <svg
                      class="fill-gray-300 dark:fill-gray-700"
                      width="8"
                      height="5"
                      viewBox="0 0 8 5"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                        d="M4.40962 4.41483C4.21057 4.69919 3.78943 4.69919 3.59038 4.41483L1.05071 0.786732C0.81874 0.455343 1.05582 0 1.46033 0H6.53967C6.94418 0 7.18126 0.455342 6.94929 0.786731L4.40962 4.41483Z"
                        fill=""
                    />
                  </svg>
                </span>
            </div>
          </th>
          <!--- 上报水位 --->
          <th class="px-4 py-3 text-left border border-gray-100 dark:border-gray-800">
            <div
                class="flex items-center justify-between w-full cursor-pointer"
                @click="sortBy('waterlevel')"
            >
              <p class="font-medium text-gray-700 text-theme-xs dark:text-gray-400">上报水位</p>
              <span class="flex flex-col gap-0.5">
                  <svg
                      class="fill-gray-300 dark:fill-gray-700"
                      width="8"
                      height="5"
                      viewBox="0 0 8 5"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                        d="M4.40962 0.585167C4.21057 0.300808 3.78943 0.300807 3.59038 0.585166L1.05071 4.21327C0.81874 4.54466 1.05582 5 1.46033 5H6.53967C6.94418 5 7.18126 4.54466 6.94929 4.21327L4.40962 0.585167Z"
                        fill=""
                    />
                  </svg>
                  <svg
                      class="fill-gray-300 dark:fill-gray-700"
                      width="8"
                      height="5"
                      viewBox="0 0 8 5"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                        d="M4.40962 4.41483C4.21057 4.69919 3.78943 4.69919 3.59038 4.41483L1.05071 0.786732C0.81874 0.455343 1.05582 0 1.46033 0H6.53967C6.94418 0 7.18126 0.455342 6.94929 0.786731L4.40962 4.41483Z"
                        fill=""
                    />
                  </svg>
                </span>
            </div>
          </th>
          <!--- 告警状态 --->
          <th class="px-4 py-3 text-left border border-gray-100 dark:border-gray-800">
            <div
                class="flex items-center justify-between w-full cursor-pointer"
                @click="sortBy('status')"
            >
              <p class="font-medium text-gray-700 text-theme-xs dark:text-gray-400">Status</p>
              <span class="flex flex-col gap-0.5">
                  <svg
                      class="fill-gray-300 dark:fill-gray-700"
                      width="8"
                      height="5"
                      viewBox="0 0 8 5"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                        d="M4.40962 0.585167C4.21057 0.300808 3.78943 0.300807 3.59038 0.585166L1.05071 4.21327C0.81874 4.54466 1.05582 5 1.46033 5H6.53967C6.94418 5 7.18126 4.54466 6.94929 4.21327L4.40962 0.585167Z"
                        fill=""
                    />
                  </svg>
                  <svg
                      class="fill-gray-300 dark:fill-gray-700"
                      width="8"
                      height="5"
                      viewBox="0 0 8 5"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                        d="M4.40962 4.41483C4.21057 4.69919 3.78943 4.69919 3.59038 4.41483L1.05071 0.786732C0.81874 0.455343 1.05582 0 1.46033 0H6.53967C6.94418 0 7.18126 0.455342 6.94929 0.786731L4.40962 4.41483Z"
                        fill=""
                    />
                  </svg>
                </span>
            </div>
          </th>
        </tr>
        </thead>
        <tbody>
        <tr
            v-for="station in paginatedData"
            :key="station.id"
            class=""
        >
          <td class="px-4 py-3 border border-gray-100 dark:border-gray-800">
            <a class="block font-medium text-gray-800 text-theme-sm dark:text-white/90"
               :href="`/station/${station.id}`">
              {{ station.id }}
            </a>
          </td>
          <td class="px-4 py-3 border border-gray-100 dark:border-gray-800">
            <a class="block font-medium text-gray-800 text-theme-sm dark:text-white/90"
               :href="`/station/${station.id}`">
              {{ station.name }}
            </a>
          </td>

          <td class="px-4 py-3 border border-gray-100 dark:border-gray-800">
            <p class="text-gray-700 text-theme-sm dark:text-gray-400">{{ station.flood_limit }}</p>
          </td>
          <td class="px-4 py-3 border border-gray-100 dark:border-gray-800">
            <p class="text-gray-700 text-theme-sm dark:text-gray-400">{{ station.warning }}</p>
          </td>
          <td class="px-4 py-3 border border-gray-100 dark:border-gray-800">
            <p class="text-gray-700 text-theme-sm dark:text-gray-400">{{ station.guaranteed }}</p>
          </td>
          <td class="px-4 py-3 border border-gray-100 dark:border-gray-800">
            <p class="text-gray-700 text-theme-sm dark:text-gray-400">{{ station.time }}</p>
          </td>
          <td class="px-4 py-3 border border-gray-100 dark:border-gray-800">
            <p class="text-gray-700 text-theme-sm dark:text-gray-400">{{ station.waterlevel }}</p>
          </td>

          <td class="px-4 py-3 border border-gray-100 dark:border-gray-800">
              <span
                  :class="{
                  'bg-success-50 dark:bg-success-500/15 text-success-700 dark:text-success-500':
                    station.status === 0,
                  'bg-error-50 dark:bg-error-500/15 text-error-700 dark:text-error-500':
                    station.status === 1,
                }"
                  class="rounded-full px-2 py-0.5 text-theme-xs font-medium"
              >
                {{ station.status ? '告警中' : '正常' }}
              </span>
          </td>
        </tr>
        </tbody>
      </table>
    </div>

    <!-- Pagination Controls -->
    <div
        class="border border-t-0 rounded-b-xl border-gray-100 py-4 pl-[18px] pr-4 dark:border-gray-800"
    >
      <div class="flex flex-col xl:flex-row xl:items-center xl:justify-between">
        <p
            class="pb-3 text-sm font-medium text-center text-gray-500 border-b border-gray-100 dark:border-gray-800 dark:text-gray-400 xl:border-b-0 xl:pb-0 xl:text-left"
        >
          显示第 {{ startEntry }} 到 {{ endEntry }} 条，共 {{ totalEntries }} 条记录
        </p>
        <div class="flex items-center justify-center gap-0.5 pt-3 xl:justify-end xl:pt-0">
          <button
              @click="prevPage"
              :disabled="currentPage === 1"
              class="mr-2.5 flex items-center h-10 justify-center rounded-lg border border-gray-300 bg-white px-3.5 py-2.5 text-gray-700 shadow-theme-xs hover:bg-gray-50 disabled:opacity-50 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-400 dark:hover:bg-white/[0.03]"
          >
            上一页
          </button>
          <button
              @click="goToPage(1)"
              :class="
              currentPage === 1
                ? 'bg-blue-500/[0.08] text-brand-500'
                : 'text-gray-700 dark:text-gray-400'
            "
              class="flex h-10 w-10 items-center justify-center rounded-lg text-sm font-medium hover:bg-blue-500/[0.08] hover:text-brand-500 dark:hover:text-brand-500"
          >
            1
          </button>
          <span
              v-if="currentPage > 3"
              class="flex h-10 w-10 items-center justify-center rounded-lg hover:bg-blue-500/[0.08] hover:text-brand-500 dark:hover:text-brand-500"
          >...</span
          >
          <button
              v-for="page in pagesAroundCurrent"
              :key="page"
              @click="goToPage(page)"
              :class="
              currentPage === page
                ? 'bg-blue-500/[0.08] text-brand-500'
                : 'text-gray-700 dark:text-gray-400'
            "
              class="flex h-10 w-10 items-center justify-center rounded-lg text-sm font-medium hover:bg-blue-500/[0.08] hover:text-brand-500 dark:hover:text-brand-500"
          >
            {{ page }}
          </button>
          <span
              v-if="currentPage < totalPages - 2"
              class="flex h-10 w-10 items-center justify-center rounded-lg text-sm font-medium text-gray-700 hover:bg-blue-500/[0.08] hover:text-brand-500 dark:text-gray-400 dark:hover:text-brand-500"
          >...</span
          >
          <button
              v-if="totalPages > 1"
              @click="goToPage(totalPages)"
              :class="
              currentPage === totalPages
                ? 'bg-blue-500/[0.08] text-brand-500'
                : 'text-gray-700 dark:text-gray-400'
            "
              class="flex h-10 w-10 items-center justify-center rounded-lg text-sm font-medium hover:bg-blue-500/[0.08] hover:text-brand-500 dark:hover:text-brand-500"
          >
            {{ totalPages }}
          </button>
          <button
              @click="nextPage"
              :disabled="currentPage === totalPages"
              class="ml-2.5 flex items-center h-10 justify-center rounded-lg border border-gray-300 bg-white px-3.5 py-2.5 text-gray-700 shadow-theme-xs hover:bg-gray-50 disabled:opacity-50 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-400 dark:hover:bg-white/[0.03]"
          >
            下一页
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import {ref, computed, onMounted} from 'vue'
import axios from "axios";

const search = ref('')
const sortColumn = ref('status')
const sortDirection = ref('desc')
const currentPage = ref(1)
const perPage = ref(10)
const {county} = defineProps({
  county: {
    type: String,
    required: true
  }
})
console.log(county)
/**
 * @typedef {Object} Station
 * @property {string} id
 * @property {string} name
 * @property {string} position
 * @property {number|null} flood_limit
 * @property {number|null} guaranteed
 * @property {number|null} warning
 * @property {Date} time
 * @property {number} waterlevel
 * @property {number} status
 */

/** @type {import('vue').Ref<Station[]>} */
const data = ref([])

async function fetchdata() {
  try {
    const response = await axios.get('/api/stationlist/', {
      params:{
        county: county
      }
    })
    for (let station of response.data) {
      station.time = new Date(station.time).toLocaleString()
      data.value.push(station)
    }
  } catch (error) {
    console.log(error)

  }
}

const filteredData = computed(() => {
  const searchLower = String(search.value).toLowerCase()
  return data.value
      .filter(
          (station) =>
              String(station.id).toLowerCase().includes(searchLower) ||
              station.name.toLowerCase().includes(searchLower) ||
              station.position.toLowerCase().includes(searchLower)
      )
      .sort((a, b) => {
        let modifier = sortDirection.value === 'asc' ? 1 : -1
        if (a[sortColumn.value] < b[sortColumn.value]) return -1 * modifier
        if (a[sortColumn.value] > b[sortColumn.value]) return 1 * modifier
        return 0
      })
})

const paginatedData = computed(() => {
  const start = (currentPage.value - 1) * perPage.value
  const end = start + perPage.value
  return filteredData.value.slice(start, end)
})

const totalEntries = computed(() => filteredData.value.length)
const startEntry = computed(() => (currentPage.value - 1) * perPage.value + 1)
const endEntry = computed(() => {
  const end = currentPage.value * perPage.value
  return end > totalEntries.value ? totalEntries.value : end
})
const totalPages = computed(() => Math.ceil(filteredData.value.length / perPage.value))

const pagesAroundCurrent = computed(() => {
  let pages = []
  const startPage = Math.max(2, currentPage.value - 2)
  const endPage = Math.min(totalPages.value - 1, currentPage.value + 2)

  for (let i = startPage; i <= endPage; i++) {
    pages.push(i)
  }
  return pages
})

const goToPage = (page) => {
  if (page >= 1 && page <= totalPages.value) {
    currentPage.value = page
  }
}

const nextPage = () => {
  if (currentPage.value < totalPages.value) {
    currentPage.value++
  }
}

const prevPage = () => {
  if (currentPage.value > 1) {
    currentPage.value--
  }
}

const sortBy = (column) => {
  if (sortColumn.value === column) {
    sortDirection.value = sortDirection.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortDirection.value = 'asc'
    sortColumn.value = column
  }
}

onMounted(() => {
  fetchdata()
})
</script>

<style scoped>
</style>
