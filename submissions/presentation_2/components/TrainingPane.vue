<script setup lang="ts">
// Using UnoCSS to style the component
import { ref, onMounted, watch } from 'vue'
import { SliderRange, SliderRoot, SliderThumb, SliderTrack } from 'radix-vue'

interface Props {
  gt: string;
  videos: string[];
  labels: string[];
  infoBox: string;
  infoLabel: string;
  size: number;
}

const numFrames = 199;
const frameRate = 10;


const props = defineProps<Props>();
const divWidth = ref(props.size ?? 120);
const videoRefs = ref<HTMLVideoElement[]>([]);
const sliderValue = ref([0])
const playing = ref(false)

const togglePlay = () => {
  playing.value = !playing.value;
  videoRefs.value.forEach(video => {
    if (playing.value) {
      video.play();
    } else {
      video.pause();
    }
  });
};

const updateSliderValue = () => {
  if (videoRefs.value.length > 0) {
    const currentTime = videoRefs.value[0].currentTime;
    sliderValue.value = [Math.floor(currentTime * frameRate)];
  }
};

const updateVideoTime = () => {
  const newTime = sliderValue.value[0] / frameRate;
  videoRefs.value.forEach(video => {
    video.currentTime = newTime;
  });
};

onMounted(() => {
  console.log(props.videos);
  console.log(videoRefs.value); // Access the video elements here

  // update initial video time

  videoRefs.value.forEach(video => {
    video.addEventListener('timeupdate', updateSliderValue);
  });
});

watch(sliderValue, () => !playing.value && updateVideoTime());

</script>
<template>
  <div class="absolute top-2.5em right-2.5em">
    <img src="/comparison_0/colorbar.png" class="w-350px" alt="Colorbar" />
  </div>
  <div class="flex flex-col flex-1 gap-[1em] justify-between">
    <div class="flex flex-row gap-[2em] justify-start items-start">

      <slot name="left-pane"></slot>

      <!-- Start with the Ground Truth Label, simple image -->
      <div class="flex flex-col gap-[1em] justify-center items-start  flex-grow-0 flex-shrink-0">
        <span class="text-white text-center">Ground Truth</span>
        <img :src="props.gt" class="flex-grow-0 flex-shrink-0 image-render-pixel"
          :style="{ width: (divWidth * 0.87) + 'px' }" />
      </div>
      <!-- Loop through the videos (can only be videos) -->
      <div v-for="(video, index) in videos" :key="index"
        class="flex-grow-0 flex-shrink-0 flex flex-col gap-[1em] justify-center items-start"
        :style="{ width: divWidth }">
        <span class="text-white text-center flex-grow-0 flex-shrink-0">{{ labels[index] }}</span>
        <video ref="videoRefs" :style="{ width: divWidth + 'px' }" muted playsinline>
          <source :src="video" type="video/mp4" />
        </video>
      </div>
      <!-- Show the info box -->
      <div v-if="props.infoBox"
        class="rounded-2 border-gray-200 border bg-white bg-op-10 text-black p-[1em] rounded-[10px] -m-4 p-4 flex flex-col gap-[1em] justify-center items-start">
        <span class="text-white text-center">{{ infoLabel }}</span>
        <img :src="props.infoBox" :style="{ width: divWidth + 'px' }" class="max-w-none" />
      </div>
    </div>
    <div v-if="counterMargin" class="h-[1em]"></div>
    <div class="w-[100%] flex gap-[1em] items-center">
      <button @click="togglePlay" class="w-10 h-10 bg-white text-black rounded-full flex items-center justify-center">
        <svg v-if="playing" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"
          fill="currentColor" class="icon icon-tabler icons-tabler-filled icon-tabler-player-pause">
          <path stroke="none" d="M0 0h24v24H0z" fill="none" />
          <path d="M9 4h-2a2 2 0 0 0 -2 2v12a2 2 0 0 0 2 2h2a2 2 0 0 0 2 -2v-12a2 2 0 0 0 -2 -2z" />
          <path d="M17 4h-2a2 2 0 0 0 -2 2v12a2 2 0 0 0 2 2h2a2 2 0 0 0 2 -2v-12a2 2 0 0 0 -2 -2z" />
        </svg>
        <svg v-else xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor"
          class="icon icon-tabler icons-tabler-filled icon-tabler-player-play">
          <path stroke="none" d="M0 0h24v24H0z" fill="none" />
          <path d="M6 4v16a1 1 0 0 0 1.524 .852l13 -8a1 1 0 0 0 0 -1.704l-13 -8a1 1 0 0 0 -1.524 .852z" />
        </svg>

      </button>
      <SliderRoot :disabled="playing" v-model="sliderValue"
        class="group relative flex items-center select-none touch-none flex-1 h-5" :max="numFrames" :step="1">
        <SliderTrack class="bg-white bg-op-60 relative grow rounded-full h-[3px] group-disabled:bg-op-30">
          <SliderRange class="absolute bg-white rounded-full h-full group-disabled:bg-op-30" />
        </SliderTrack>
        <SliderThumb
          class="block w-5 h-5 bg-white rounded-[10px] hover:bg-gray-200 focus:outline-none focus:shadow-[0_0_0_5px] focus:shadow-blackA8"
          aria-label="Volume" />
      </SliderRoot>
      <!-- Show the value of the epoch -->
      <span class="text-white">Epoch: {{ sliderValue[0] }}</span>
    </div>
  </div>
</template>