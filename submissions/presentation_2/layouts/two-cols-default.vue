<!-- default.vue -->
<script setup lang="ts">
const props = defineProps({
  class: {
    type: String,
  },
  layoutClass: {
    type: String,
  },
})
</script>


<template>
  <div class="slidev-layout default flex flex-col gap-[2em]">
    <div>
      <slot />
    </div>
    <div class="flex-1 grid grid-cols-2 gap-[2em]">
      <div class="col-span-1" :class="props.class">
        <slot name="left" />
      </div>
      <div class="col-span-1" :class="props.class">
        <slot name="right" />
      </div>
    </div>
  </div>
</template>