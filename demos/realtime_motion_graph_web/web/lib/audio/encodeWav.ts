// Encode an AudioBuffer as 16-bit PCM WAV. Channels are interleaved.
// Float samples are clamped to [-1, 1] and scaled to Int16 range.
export function encodeWav(buffer: AudioBuffer): Blob {
  const numCh = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const numFrames = buffer.length;
  const bytesPerSample = 2;
  const dataLen = numFrames * numCh * bytesPerSample;
  const bufferSize = 44 + dataLen;

  const out = new ArrayBuffer(bufferSize);
  const view = new DataView(out);

  // RIFF header
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataLen, true);
  writeAscii(view, 8, "WAVE");

  // fmt subchunk (PCM)
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true); // subchunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, numCh, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numCh * bytesPerSample, true); // byte rate
  view.setUint16(32, numCh * bytesPerSample, true); // block align
  view.setUint16(34, 8 * bytesPerSample, true); // bits per sample

  // data subchunk
  writeAscii(view, 36, "data");
  view.setUint32(40, dataLen, true);

  // Pre-pull channel data — getChannelData() can be expensive if called per-sample.
  const channels: Float32Array[] = new Array(numCh);
  for (let c = 0; c < numCh; c++) channels[c] = buffer.getChannelData(c);

  // Interleave + convert. Use a typed-array view onto the existing buffer so
  // we don't allocate a second copy.
  const pcm = new Int16Array(out, 44, numFrames * numCh);
  let p = 0;
  for (let i = 0; i < numFrames; i++) {
    for (let c = 0; c < numCh; c++) {
      const s = channels[c][i];
      const clamped = s < -1 ? -1 : s > 1 ? 1 : s;
      // Asymmetric scaling (32767 / 32768) keeps full negative range without wrapping.
      pcm[p++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    }
  }

  return new Blob([out], { type: "audio/wav" });
}

function writeAscii(view: DataView, offset: number, text: string) {
  for (let i = 0; i < text.length; i++) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}
