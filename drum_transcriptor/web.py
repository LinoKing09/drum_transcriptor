"""Small local web UI: paste a YouTube link, get drum sheet music."""

from __future__ import annotations

import tempfile
import threading
import uuid
import webbrowser
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file

from . import pipeline

app = Flask(__name__)
OUT = Path(tempfile.gettempdir()) / "drum_transcriptor_web"
JOBS: dict[str, dict] = {}


def _work(job_id: str, url: str, opts: pipeline.Options) -> None:
    job = JOBS[job_id]

    def progress(msg: str, frac: float) -> None:
        job.update(message=msg, progress=frac)

    try:
        res = pipeline.run(url, OUT / job_id, opts, progress)
        job.update(state="done", title=res.title, tempo=res.tempo, bars=res.bars,
                   files={"pdf": res.pdf, "midi": res.midi, "ly": res.ly})
    except Exception as exc:  # noqa: BLE001 - reported to the browser
        job.update(state="error", message=str(exc))


@app.post("/api/jobs")
def create_job():
    data = request.get_json(force=True)
    url = (data.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return jsonify(error="Bitte einen gültigen YouTube-Link eingeben."), 400

    def num(key):
        value = data.get(key)
        return float(value) if value not in (None, "") else None

    opts = pipeline.Options(
        sensitivity=num("sensitivity") or 1.0,
        bpm=num("bpm"),
        beats_per_bar=int(data.get("beats_per_bar") or 4),
        triplets=bool(data.get("triplets", True)),
        start=num("start"),
        end=num("end"),
    )
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"state": "running", "message": "Starte …", "progress": 0.0}
    threading.Thread(target=_work, args=(job_id, url, opts), daemon=True).start()
    return jsonify(id=job_id)


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    job = JOBS.get(job_id) or abort(404)
    return jsonify({k: v for k, v in job.items() if k != "files"})


@app.get("/api/jobs/<job_id>/<kind>")
def job_file(job_id: str, kind: str):
    job = JOBS.get(job_id) or abort(404)
    path = job.get("files", {}).get(kind) or abort(404)
    return send_file(path, as_attachment=request.args.get("download") == "1",
                     download_name=path.name)


@app.get("/")
def index():
    return PAGE


def serve(port: int = 8000, open_browser: bool = True) -> None:
    url = f"http://127.0.0.1:{port}"
    print(f"🥁  Drum Transcriptor läuft auf {url}  (Strg+C zum Beenden)")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, debug=False)


PAGE = """<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drum Transcriptor</title>
<style>
  :root {
    --bg: #f6f4ef; --card: #ffffff; --ink: #1d1b18; --muted: #6b665e;
    --accent: #d1462f; --accent-ink: #ffffff; --line: #e3ded4; --track: #ece7dd;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #161513; --card: #211f1c; --ink: #f1ede6; --muted: #a39d93;
      --accent: #ef6a52; --accent-ink: #1b0f0c; --line: #34312c; --track: #2c2925;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 16px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 760px; margin: 0 auto; padding: 48px 16px 64px; }
  h1 { font-size: 2rem; margin: 0 0 4px; letter-spacing: -0.02em; }
  h1 span { color: var(--accent); }
  .sub { color: var(--muted); margin: 0 0 28px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 14px;
          padding: 20px; }
  .row { display: flex; gap: 10px; }
  input, select { font: inherit; color: var(--ink); background: var(--bg);
                  border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px;
                  width: 100%; min-width: 0; }
  input:focus, select:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
  button { font: inherit; font-weight: 600; border: 0; border-radius: 10px; cursor: pointer;
           padding: 10px 18px; background: var(--accent); color: var(--accent-ink);
           white-space: nowrap; }
  button:disabled { opacity: .5; cursor: default; }
  details { margin-top: 14px; }
  summary { cursor: pointer; color: var(--muted); }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 12px; margin-top: 12px; }
  label { font-size: .85rem; color: var(--muted); display: block; }
  label input, label select { margin-top: 4px; }
  .check { display: flex; align-items: center; gap: 8px; margin-top: 22px; color: var(--ink); }
  .check input { width: auto; }
  #status { margin-top: 20px; display: none; }
  .bar { height: 8px; background: var(--track); border-radius: 99px; overflow: hidden; }
  .bar div { height: 100%; width: 0; background: var(--accent); transition: width .4s; }
  .msg { color: var(--muted); margin-top: 8px; font-size: .95rem; }
  .err { color: var(--accent); }
  #result { margin-top: 20px; display: none; }
  #result h2 { margin: 0 0 2px; font-size: 1.25rem; }
  .meta { color: var(--muted); margin-bottom: 14px; }
  .links { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }
  .links a { text-decoration: none; color: var(--ink); border: 1px solid var(--line);
             border-radius: 10px; padding: 8px 14px; background: var(--card); }
  .links a.primary { background: var(--accent); color: var(--accent-ink); border-color: transparent;
                     font-weight: 600; }
  iframe { width: 100%; height: 80vh; border: 1px solid var(--line); border-radius: 12px;
           background: #fff; }
  .note { color: var(--muted); font-size: .85rem; margin-top: 28px; }
  @media (max-width: 520px) { .row { flex-direction: column; } }
</style>
</head>
<body>
<main>
  <h1>Drum <span>Transcriptor</span></h1>
  <p class="sub">YouTube-Link rein, Schlagzeug-Noten als PDF raus.</p>

  <form class="card" id="form">
    <div class="row">
      <input id="url" type="url" required placeholder="https://www.youtube.com/watch?v=…">
      <button id="go" type="submit">Transkribieren</button>
    </div>
    <details>
      <summary>Optionen</summary>
      <div class="grid">
        <label>Start (Sek. oder m:ss)<input id="start" placeholder="0:00"></label>
        <label>Ende (Sek. oder m:ss)<input id="end" placeholder="Songende"></label>
        <label>Tempo (BPM)<input id="bpm" type="number" min="40" max="260" placeholder="automatisch"></label>
        <label>Taktart
          <select id="beats"><option value="4">4/4</option><option value="3">3/4</option>
            <option value="2">2/4</option><option value="5">5/4</option><option value="6">6/4</option></select>
        </label>
        <label>Empfindlichkeit
          <select id="sens"><option value="0.7">niedrig</option><option value="1" selected>normal</option>
            <option value="1.4">hoch</option><option value="2">sehr hoch</option></select>
        </label>
        <label class="check"><input id="trip" type="checkbox" checked> Triolen erkennen</label>
      </div>
    </details>
    <div id="status">
      <div class="bar"><div id="fill"></div></div>
      <div class="msg" id="msg"></div>
    </div>
  </form>

  <section id="result">
    <h2 id="title"></h2>
    <div class="meta" id="meta"></div>
    <div class="links">
      <a class="primary" id="dl-pdf" href="#">PDF herunterladen</a>
      <a id="dl-midi" href="#">MIDI</a>
      <a id="dl-ly" href="#">LilyPond-Quelltext</a>
    </div>
    <iframe id="pdf" title="Noten"></iframe>
  </section>

  <p class="note">Die Erkennung ist automatisch und nicht perfekt – am besten mit dem Original
  gegenhören. Die Trennung des Schlagzeugs (Demucs) dauert je nach Rechner ein paar Minuten.</p>
</main>
<script>
const $ = id => document.getElementById(id);
const secs = v => {
  v = v.trim(); if (!v) return null;
  return v.split(':').reduce((t, p) => t * 60 + parseFloat(p), 0);
};
$('form').addEventListener('submit', async e => {
  e.preventDefault();
  $('go').disabled = true; $('result').style.display = 'none';
  $('status').style.display = 'block'; $('msg').className = 'msg';
  $('msg').textContent = 'Starte …'; $('fill').style.width = '2%';
  const body = {
    url: $('url').value, start: secs($('start').value), end: secs($('end').value),
    bpm: $('bpm').value || null, beats_per_bar: $('beats').value,
    sensitivity: $('sens').value, triplets: $('trip').checked,
  };
  try {
    const r = await fetch('/api/jobs', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                        body: JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || 'Fehler');
    poll(j.id);
  } catch (err) { fail(err.message); }
});
function fail(text) {
  $('msg').textContent = text; $('msg').className = 'msg err'; $('go').disabled = false;
}
async function poll(id) {
  const j = await (await fetch('/api/jobs/' + id)).json();
  $('fill').style.width = Math.max(2, j.progress * 100) + '%';
  $('msg').textContent = j.message;
  if (j.state === 'running') return setTimeout(() => poll(id), 1000);
  if (j.state === 'error') return fail(j.message);
  $('go').disabled = false;
  const base = '/api/jobs/' + id + '/';
  $('title').textContent = j.title;
  $('meta').textContent = j.bars + ' Takte · ♩ = ' + j.tempo;
  $('dl-pdf').href = base + 'pdf?download=1';
  $('dl-midi').href = base + 'midi?download=1';
  $('dl-ly').href = base + 'ly?download=1';
  $('pdf').src = base + 'pdf';
  $('result').style.display = 'block';
}
</script>
</body>
</html>
"""
