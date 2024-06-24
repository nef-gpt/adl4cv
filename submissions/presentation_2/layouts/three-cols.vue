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
  <div class="slidev-layout default">
    <slot />
    <div class=" grid grid-cols-3 gap-[2em]">
      <div class="col-span-1" :class="props.class">
        <slot name="left" />
      </div>
      <div class="col-span-1" :class="props.class">
        <slot name="center" />
      </div>
      <div class="col-span-1" :class="props.class">
        <slot name="right" />
      </div>
    </div>
  </div>
</template>