<script setup lang="ts">
// Using UnoCSS to style the component
import { ref, onMounted, watch } from 'vue'
import { SliderRange, SliderRoot, SliderThumb, SliderTrack } from 'radix-vue'

interface Props {
  videos: string[];
  rowLabels: string[];
}


const numFrames = 14_000 / 100;
const frameRate = 10;

const props = defineProps<Props>();
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
  <div class="flex flex-col gap-[1em] items-start">
    <div v-for="(row, index) in videos" :key="index" class="flex flex-row gap-[1em] items-center">
      <span v-if="props.rowLabels && props.rowLabels.length > 0" class="text-white w-100px">{{ props.rowLabels[index]
        }}</span>
      <div v-for="(video, index) in row" :key="index" class="video-pane">
        <video ref="videoRefs" class="w-[100px]" muted playsinline loop>
          <source :src="video" type="video/mp4" />
        </video>
      </div>
    </div>
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