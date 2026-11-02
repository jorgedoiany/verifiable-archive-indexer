// Cue the video at the cited range and add keyboard shortcuts for the judgments.
(function () {
  const player = document.getElementById("player");
  if (!player) return;
  const start = Number(player.dataset.start);
  const end = Number(player.dataset.end);
  const hasRange =
    player.dataset.start !== undefined && Number.isFinite(start) && Number.isFinite(end);
  let stopAt = null;

  function playRange() {
    if (!hasRange) {
      player.currentTime = 0;
    } else {
      player.currentTime = Math.max(0, start - 1);
      stopAt = end + 1;
    }
    player.play().catch(() => {});
  }

  player.addEventListener("loadedmetadata", () => {
    player.currentTime = hasRange ? Math.max(0, start - 1) : 0;
  });
  player.addEventListener("timeupdate", () => {
    if (stopAt !== null && player.currentTime >= stopAt) {
      player.pause();
      stopAt = null;
    }
  });

  const playButton = document.getElementById("play-range");
  if (playButton) playButton.addEventListener("click", playRange);

  const shortcuts = { "1": "exists", "2": "not_exists", "3": "unsure" };
  document.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const tag = event.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    if (event.key === "r" || event.key === "R") {
      playRange();
      return;
    }
    const value = shortcuts[event.key];
    if (!value) return;
    const button = document.querySelector(
      'button[name="verdict"][value="' + value + '"]'
    );
    if (button) button.click();
  });
})();
