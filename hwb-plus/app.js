const comparisonList = document.querySelector('#comparison-list');
const comparisonTemplate = document.querySelector('#comparison-template');
const variantTemplate = document.querySelector('#variant-template');
const count = document.querySelector('#sample-count');
const filterButtons = [...document.querySelectorAll('.filter-button')];

let comparisons = [];
let activeFilter = 'all';

function formatDuration(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remaining = Math.round(seconds % 60).toString().padStart(2, '0');
  return `${minutes}:${remaining}`;
}

function renderComparisons() {
  const visibleComparisons = activeFilter === 'all'
    ? comparisons
    : comparisons.filter((comparison) => comparison.speaker === activeFilter);

  comparisonList.replaceChildren();
  count.textContent = `${visibleComparisons.length} ${visibleComparisons.length === 1 ? 'utterance' : 'utterances'}`;

  visibleComparisons.forEach((comparison) => {
    const node = comparisonTemplate.content.cloneNode(true);
    const title = `${comparison.speaker} · ${comparison.utterance} · ${comparison.microphone}`;
    const columns = node.querySelector('.comparison-columns');

    node.querySelector('.comparison-id').textContent = comparison.id;
    node.querySelector('.comparison-title').textContent = title;

    comparison.variants.forEach((variant) => {
      const variantNode = variantTemplate.content.cloneNode(true);
      const image = variantNode.querySelector('.spectrogram');
      const audio = variantNode.querySelector('.audio-player');

      variantNode.querySelector('.variant-label').textContent = variant.label;
      variantNode.querySelector('.variant-detail').textContent = variant.detail;
      image.src = variant.spectrogram;
      image.alt = `${variant.label} spectrogram for ${title}`;
      audio.src = variant.audio;
      audio.setAttribute('aria-label', `Play ${variant.label} for ${title}`);
      variantNode.querySelector('.variant-meta').textContent = `${formatDuration(variant.duration)} · ${(variant.sampleRate / 1000).toFixed(2)} kHz`;

      audio.addEventListener('play', () => {
        document.querySelectorAll('audio').forEach((otherAudio) => {
          if (otherAudio !== audio) otherAudio.pause();
        });
      });
      columns.append(variantNode);
    });

    comparisonList.append(node);
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
    renderComparisons();
  });
});

fetch('comparisons.json')
  .then((response) => {
    if (!response.ok) throw new Error('Unable to load audio comparisons.');
    return response.json();
  })
  .then((payload) => {
    comparisons = payload.comparisons;
    renderComparisons();
  })
  .catch((error) => {
    count.textContent = error.message;
  });
