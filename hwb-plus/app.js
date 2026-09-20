const grid = document.querySelector('#sample-grid');
const template = document.querySelector('#sample-template');
const count = document.querySelector('#sample-count');
const filterButtons = [...document.querySelectorAll('.filter-button')];

let samples = [];
let activeFilter = 'all';

function formatDuration(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remaining = Math.round(seconds % 60).toString().padStart(2, '0');
  return `${minutes}:${remaining}`;
}

function renderSamples() {
  const visibleSamples = activeFilter === 'all'
    ? samples
    : samples.filter((sample) => sample.speaker === activeFilter);

  grid.replaceChildren();
  count.textContent = `${visibleSamples.length} ${visibleSamples.length === 1 ? 'sample' : 'samples'}`;

  visibleSamples.forEach((sample) => {
    const node = template.content.cloneNode(true);
    const title = `${sample.speaker} · ${sample.utterance} · ${sample.microphone}`;
    const role = node.querySelector('.sample-role');
    const image = node.querySelector('.spectrogram');
    const audio = node.querySelector('.audio-player');

    node.querySelector('.sample-id').textContent = sample.id;
    node.querySelector('.sample-title').textContent = title;
    role.textContent = sample.role;
    role.classList.toggle('full-band', sample.role === 'Full-band sample');
    image.src = sample.spectrogram;
    image.alt = `Spectrogram for ${title}`;
    audio.src = sample.audio;
    audio.setAttribute('aria-label', `Play ${title}`);
    node.querySelector('.duration').textContent = formatDuration(sample.duration);
    node.querySelector('.sample-rate').textContent = `${(sample.sampleRate / 1000).toFixed(1)} kHz`;
    node.querySelector('.channels').textContent = sample.channels === 1 ? 'Mono' : `${sample.channels} ch`;

    audio.addEventListener('play', () => {
      document.querySelectorAll('audio').forEach((otherAudio) => {
        if (otherAudio !== audio) otherAudio.pause();
      });
    });
    grid.append(node);
  });
}

filterButtons.forEach((button) => {
  button.addEventListener('click', () => {
    activeFilter = button.dataset.filter;
    filterButtons.forEach((item) => {
      const isActive = item === button;
      item.classList.toggle('active', isActive);
      item.setAttribute('aria-pressed', String(isActive));
    });
    renderSamples();
  });
});

fetch('samples.json')
  .then((response) => {
    if (!response.ok) throw new Error('Unable to load audio examples.');
    return response.json();
  })
  .then((payload) => {
    samples = payload.samples;
    renderSamples();
  })
  .catch((error) => {
    count.textContent = error.message;
  });
