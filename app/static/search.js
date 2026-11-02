// Play a result's video from the moment its evidence points at.
(function () {
  const player = document.getElementById("player");
  if (!player) return;
  const label = document.getElementById("now-playing");
  let pendingSeek = null;

  player.addEventListener("loadedmetadata", () => {
    if (pendingSeek === null) return;
    player.currentTime = pendingSeek;
    pendingSeek = null;
    player.play().catch(() => {});
  });

  player.addEventListener("error", () => {
    pendingSeek = null;
    if (label) label.textContent = "The video could not be loaded.";
  });

  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-play]");
    if (!button) return;
    const stem = button.dataset.stem || "";
    const seconds = Number(button.dataset.at);
    if (!/^[A-Za-z0-9_-]+$/.test(stem) || !Number.isFinite(seconds)) return;
    const at = Math.max(0, seconds);
    if (label) label.textContent = "Now playing: " + stem + " from " + at.toFixed(1) + " s";
    if (player.dataset.stem === stem && player.readyState >= 1) {
      player.currentTime = at;
      player.play().catch(() => {});
      return;
    }
    player.dataset.stem = stem;
    pendingSeek = at;
    player.src = "/media/" + stem;
    player.load();
  });
})();
