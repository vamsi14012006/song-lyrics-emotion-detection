const $ = (id) => document.getElementById(id);

$("go").addEventListener("click", async () => {
  const lyrics = $("lyrics").value;
  const status = $("status"), btn = $("go");
  status.className = ""; status.textContent = "Analyzing…"; btn.disabled = true;
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lyrics }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Something went wrong.");
    status.textContent = "";
    $("top").textContent = data.emotion;
    $("bars").innerHTML = data.scores.map(s => {
      const pct = (s.probability * 100).toFixed(1);
      return `<li><span>${s.emotion}</span><div class="track"><div class="fill" data-w="${pct}"></div></div><span>${pct}%</span></li>`;
    }).join("");
    $("result").hidden = false;
    requestAnimationFrame(() => document.querySelectorAll(".fill").forEach(f => f.style.width = f.dataset.w + "%"));
  } catch (e) {
    status.className = "err"; status.textContent = e.message;
  } finally { btn.disabled = false; }
});
