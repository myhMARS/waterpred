<template>
  <div class="bg-white">
    <Header/>
    <div class="relative isolate overflow-hidden bg-linear-to-b from-indigo-100/20 pt-14">
      <div
          class="absolute inset-y-0 right-1/2 -z-10 -mr-96 w-[200%] origin-top-right skew-x-[-30deg] bg-white ring-1 shadow-xl shadow-indigo-600/10 ring-indigo-50 sm:-mr-80 lg:-mr-96"
          aria-hidden="true"/>
      <div class="mx-auto max-w-7xl px-6 py-32 sm:py-40 lg:px-8">
        <div
            class="mx-auto max-w-2xl lg:mx-0 lg:grid lg:max-w-none lg:grid-cols-2 lg:gap-x-16 lg:gap-y-8 xl:grid-cols-1 xl:grid-rows-1 xl:gap-x-8">
          <h1
              class="max-w-2xl text-5xl font-semibold tracking-tight text-balance text-gray-900 sm:text-7xl lg:col-span-2 xl:col-auto">
            基于LSTM的智能洪水预测与预警系统</h1>
          <div class="mt-6 max-w-xl lg:mt-0 xl:col-end-1 xl:row-start-1">
            <p class="text-lg font-medium text-pretty text-gray-500 sm:text-xl/8">本项目旨在利用
              LSTM（长短时记忆网络）进行洪水预测，通过分析历史水文数据，
              系统能够预测该流域在不同条件下的水位变化趋势，提供精准的洪水预警信息，帮助相关部门更好地应对潜在的洪水风险，提升防灾减灾能力。
              当前示例 DEMO 主要应用于预测里畈东坑溪-里畈水库-桥东村一段流域的水位变化。</p>
            <div class="mt-10 flex items-center gap-x-6">
              <a href="/demo"
                 class="rounded-md bg-indigo-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-xs hover:bg-indigo-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600">
                开始
              </a>
              <a href="#" class="text-sm/6 font-semibold text-gray-900">更多 <span aria-hidden="true">→</span></a>
            </div>
          </div>
          <img src="@/assets/img/location.jpg" alt=""
               class="mt-8 aspect-6/5 w-full max-w-lg rounded-2xl object-cover sm:mt-16 lg:mt-0 lg:max-w-none xl:row-span-2 xl:row-end-2 xl:mt-36"/>
        </div>
      </div>
      <div class="absolute inset-x-0 bottom-0 -z-10 h-24 bg-linear-to-t from-white sm:h-32"/>
    </div>
  </div>
</template>

<script setup>
import Header from "@/components/Header.vue";
</script>
