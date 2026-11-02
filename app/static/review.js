// Seek the video to the moment an evidence item points at.
document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-seek]");
  if (!button) return;
  const player = document.getElementById("player");
  if (!player) return;
  const seconds = Number(button.dataset.seek);
  if (!Number.isFinite(seconds)) return;
  player.currentTime = Math.max(0, seconds);
  player.play().catch(() => {});
});
