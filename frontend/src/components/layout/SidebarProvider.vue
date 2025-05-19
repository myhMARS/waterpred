<template>
  <slot></slot>
</template>

<script setup lang="js">
import { useSidebarProvider } from '@/composables/useSidebar'

useSidebarProvider()
</script>
