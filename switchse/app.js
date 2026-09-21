const comparisonList = document.querySelector('#comparison-list');
const comparisonTemplate = document.querySelector('#comparison-template');
const variantTemplate = document.querySelector('#variant-template');
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
    : samples.filter((sample) => sample.group === activeFilter);

  comparisonList.replaceChildren();
  count.textContent = `${visibleSamples.length} ${visibleSamples.length === 1 ? 'utterance' : 'utterances'}`;

  visibleSamples.forEach((sample) => {
    const comparison = comparisonTemplate.content.firstElementChild.cloneNode(true);
    comparison.querySelector('.comparison-id').textContent = sample.collection;
    comparison.querySelector('.comparison-title').textContent = sample.title;

    const columns = comparison.querySelector('.comparison-columns');
    columns.style.setProperty('--column-count', sample.variants.length);
    sample.variants.forEach((variant) => {
      const item = variantTemplate.content.firstElementChild.cloneNode(true);
      item.querySelector('.variant-label').textContent = variant.label;
      item.querySelector('.variant-detail').textContent = variant.detail;

      const spectrogram = item.querySelector('.spectrogram');
      spectrogram.src = variant.spectrogram;
      spectrogram.alt = `${sample.title}: ${variant.label} log-magnitude spectrogram`;

      const audio = item.querySelector('.audio-player');
      audio.src = variant.audio;
      audio.setAttribute('aria-label', `${sample.title}: ${variant.label} audio`);

      item.querySelector('.variant-meta').textContent = `${sample.sampleRate / 1000} kHz · ${formatDuration(sample.duration)}`;
      columns.append(item);
    });

    comparisonList.append(comparison);
  });
}

filterButtons.forEach((button) => {
  button.addEventListener('click', () => {
    activeFilter = button.dataset.filter;
    filterButtons.forEach((candidate) => {
      const selected = candidate === button;
      candidate.classList.toggle('active', selected);
      candidate.setAttribute('aria-pressed', selected ? 'true' : 'false');
    });
    renderSamples();
  });
});

fetch('samples.json?v=switchse-demo-v1-20260921')
  .then((response) => {
    if (!response.ok) throw new Error(`Could not load samples (${response.status})`);
    return response.json();
  })
  .then((payload) => {
    samples = payload.samples;
    renderSamples();
  })
  .catch((error) => {
    count.textContent = `Unable to load the demonstrations: ${error.message}`;
  });
