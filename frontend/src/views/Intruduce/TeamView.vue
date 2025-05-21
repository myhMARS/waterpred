<template>
  <Header/>
  <div class="bg-white py-24 sm:py-32">
    <div class="mx-auto max-w-7xl px-6 lg:px-8">
      <div class="mx-auto max-w-2xl lg:mx-0">
        <h2 class="text-4xl font-semibold tracking-tight text-pretty text-gray-900 sm:text-5xl">Our team</h2>
        <p class="mt-6 text-lg/8 text-gray-600">
          我们是一群充满活力的人，对我们的学习充满热情。</p>
      </div>
      <ul role="list"
          class="mx-auto mt-20 grid max-w-2xl grid-cols-1 gap-x-8 gap-y-16 sm:grid-cols-2 lg:mx-0 lg:max-w-none lg:grid-cols-3">
        <li v-for="person in people" :key="person.name">
          <img class="aspect-3/2 w-full rounded-2xl object-cover" :src="person.imageUrl" alt=""/>
          <h3 class="mt-6 text-lg/8 font-semibold tracking-tight text-gray-900">{{ person.name }}</h3>
          <p class="text-base/7 text-gray-600">{{ person.role }}</p>
          <ul role="list" class="mt-6 flex gap-x-6">
            <li>
              <a :href="person.githubUrl" class="text-gray-400 hover:text-gray-500">
                <span class="sr-only">X</span>
                <svg class="size-5" aria-hidden="true" fill="currentColor" viewBox="0 0 24 24">
                  <path fill-rule="evenodd" clip-rule="evenodd"
                        d="M12 2C6.48 2 2 6.48 2 12c0 4.42 2.87 8.17 6.84 9.49.5.09.68-.22.68-.48
                       0-.24-.01-.87-.01-1.7-2.79.61-3.38-1.35-3.38-1.35-.46-1.16-1.11-1.47-1.11-1.47-.91-.63.07-.62.07-.62
                       1 .07 1.53 1.03 1.53 1.03.89 1.52 2.33 1.08 2.9.83.09-.64.35-1.08.63-1.33-2.23-.25-4.57-1.11-4.57-4.93
                       0-1.09.39-1.99 1.03-2.69-.1-.25-.45-1.27.1-2.65 0 0 .84-.27 2.75 1.02A9.64 9.64 0 0 1 12 6.84a9.65 9.65
                       0 0 1 2.5.34c1.91-1.29 2.75-1.02 2.75-1.02.55 1.38.2 2.4.1 2.65.64.7 1.03 1.6
                       1.03 2.69 0 3.83-2.35 4.68-4.58 4.93.36.31.68.91.68 1.84
                       0 1.33-.01 2.41-.01 2.74 0 .27.18.58.69.48A10.002 10.002
                       0 0 0 22 12c0-5.52-4.48-10-10-10z">
                  </path>
                </svg>

              </a>
            </li>
            <!--
            <li>
              <a :href="person.linkedinUrl" class="text-gray-400 hover:text-gray-500">
                <span class="sr-only">LinkedIn</span>
                <svg class="size-5" aria-hidden="true" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M16.338 16.338H13.67V12.16c0-.995-.017-2.277-1.387-2.277-1.39 0-1.601 1.086-1.601 2.207v4.248H8.014v-8.59h2.559v1.174h.037c.356-.675 1.227-1.387 2.526-1.387 2.703 0 3.203 1.778 3.203 4.092v4.711zM5.005 6.575a1.548 1.548 0 11-.003-3.096 1.548 1.548 0 01.003 3.096zm-1.337 9.763H6.34v-8.59H3.667v8.59zM17.668 1H2.328C1.595 1 1 1.581 1 2.298v15.403C1 18.418 1.595 19 2.328 19h15.34c.734 0 1.332-.582 1.332-1.299V2.298C19 1.581 18.402 1 17.668 1z" clip-rule="evenodd" />
                </svg>
              </a>
            </li>
            -->
          </ul>
        </li>
      </ul>
    </div>
  </div>
</template>

<script setup>
import Header from "@/components/Header.vue";

const people = [
  {
    name: '孟永豪',
    role: '后端/模型开发',
    imageUrl:
        '/images/team/Eternity.png',
    githubUrl: 'https://github.com/myhMARS',
    linkedinUrl: '#',
  },
  {
    name: '商徐炜',
    role: '前端页面开发',
    imageUrl:
        '/images/team/Aurora.png',
    linkedinUrl: '#',
  },
  {
    name: '王昊',
    role: '数据集收集',
    imageUrl: '/images/team/wanghao.png',
    linkedinUrl: '#'
  }
]
</script>