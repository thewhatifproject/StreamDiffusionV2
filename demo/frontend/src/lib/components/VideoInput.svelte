<script lang="ts">
  import 'rvfc-polyfill';
  import { onDestroy, onMount } from 'svelte';
  import {
    mediaStreamStatus,
    MediaStreamStatusEnum,
    onFrameChangeStore,
    mediaStream,
    mediaDevices
  } from '$lib/mediaStream';
  import MediaListSwitcher from './MediaListSwitcher.svelte';
  export let width = 512;
  export let height = 512;
  export let isStreaming: boolean = false;
  const size = { width, height };

  let videoEl: HTMLVideoElement;
  let canvasEl: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;
  let videoFrameCallbackId: number;

  let videoIsReady = false;

  // ajust the throttle time to your needs
  const THROTTLE = 1000 / 120;

  onMount(() => {
    ctx = canvasEl.getContext('2d') as CanvasRenderingContext2D;
    canvasEl.width = size.width;
    canvasEl.height = size.height;
  });
  onDestroy(() => {
    if (videoFrameCallbackId && videoEl) videoEl.cancelVideoFrameCallback(videoFrameCallbackId);
  });

  // Bind camera stream
  $: if (videoEl) {
    videoEl.src = '';
    videoEl.load();
    videoEl.srcObject = $mediaStream;
  }

  let lastMillis = 0;
  async function onFrameChange(now: DOMHighResTimeStamp, metadata: VideoFrameCallbackMetadata) {
    if (now - lastMillis < THROTTLE) {
      videoFrameCallbackId = videoEl.requestVideoFrameCallback(onFrameChange);
      return;
    }
    const videoWidth = videoEl.videoWidth;
    const videoHeight = videoEl.videoHeight;
    let height0 = videoHeight;
    let width0 = videoWidth;
    let x0 = 0;
    let y0 = 0;
    if (videoWidth > videoHeight) {
      width0 = videoHeight;
      x0 = (videoWidth - videoHeight) / 2;
    } else {
      height0 = videoWidth;
      y0 = (videoHeight - videoWidth) / 2;
    }
    ctx.drawImage(videoEl, x0, y0, width0, height0, 0, 0, size.width, size.height);
    const blob = await new Promise<Blob>((resolve) => {
      canvasEl.toBlob(
        (blob) => {
          resolve(blob as Blob);
        },
        'image/jpeg',
        1
      );
    });
    onFrameChangeStore.set({ blob });
    videoFrameCallbackId = videoEl.requestVideoFrameCallback(onFrameChange);
  }

  // Start frame extraction when stream is ready
  $: if ($mediaStreamStatus == MediaStreamStatusEnum.CONNECTED && videoIsReady && isStreaming) {
    videoFrameCallbackId = videoEl.requestVideoFrameCallback(onFrameChange);
  }

  $: if (!isStreaming) {
    videoIsReady = false;
    lastMillis = 0;
    if (videoEl) {
      if (videoFrameCallbackId) {
        videoEl.cancelVideoFrameCallback(videoFrameCallbackId);
        videoFrameCallbackId = 0;
      }
    }
    // Optionally clear the canvas
    if (canvasEl && ctx) {
      ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    }
    // Clear the frame store to avoid sending old frames
    onFrameChangeStore.set({ blob: new Blob() });
  }

</script>

<div class="relative mx-auto aspect-square max-w-lg self-center overflow-hidden rounded-lg border border-slate-300">
  <div class="relative z-10 aspect-square w-full object-cover">
    {#if $mediaDevices.length > 0}
      <div class="absolute bottom-0 right-0 z-10">
        <MediaListSwitcher />
      </div>
    {/if}
    <video
      class="pointer-events-none aspect-square w-full object-cover h-full"
      bind:this={videoEl}
      on:loadeddata={() => { videoIsReady = true; }}
      playsinline
      autoplay
      muted
      loop
    ></video>
    <canvas bind:this={canvasEl} class="absolute left-0 top-0 aspect-square w-full object-cover h-full"
    ></canvas>
  </div>
  <div class="absolute left-0 top-0 flex aspect-square w-full items-center justify-center">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 448" class="w-40 p-5 opacity-20">
      <path
        fill="currentColor"
        d="M224 256a128 128 0 1 0 0-256 128 128 0 1 0 0 256zm-45.7 48A178.3 178.3 0 0 0 0 482.3 29.7 29.7 0 0 0 29.7 512h388.6a29.7 29.7 0 0 0 29.7-29.7c0-98.5-79.8-178.3-178.3-178.3h-91.4z"
      />
    </svg>
  </div>
</div>
