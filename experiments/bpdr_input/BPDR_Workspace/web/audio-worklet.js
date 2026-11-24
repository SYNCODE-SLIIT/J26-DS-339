/* AudioWorklet PCM capture + browser-side WAV encoder (no build step) */
class PCMRecorderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input.length > 0) {
      const ch = input[0];
      let peak = 0;
      let sumSq = 0;
      for (let i = 0; i < ch.length; i++) {
        const v = ch[i];
        const a = Math.abs(v);
        if (a > peak) peak = a;
        sumSq += v * v;
      }
      const rms = Math.sqrt(sumSq / ch.length);
      this.port.postMessage({
        samples: new Float32Array(ch),
        peak,
        rms,
      });
    }
    return true;
  }
}

registerProcessor("pcm-recorder-processor", PCMRecorderProcessor);
