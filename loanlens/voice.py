"""Read the result aloud, in English or an Indian language.

* plain_summary()      - a short spoken-style summary built from the prediction
                         alone, so there is always something to read even when
                         Ollama is offline.
* strip_for_speech()   - removes [n] citations, **bold** and bullets.
* speak_html()         - a tiny HTML/JS player using the browser's own
                         speechSynthesis (Web Speech API): nothing is sent to
                         any server, and it works offline if the OS has the voice.

Translation itself is done by the local LLM (llm.translate).
"""
from __future__ import annotations

import json
import re

from .data import format_value

# Display name -> BCP-47 code used to pick a browser voice.
LANGUAGES = {"English": "en-IN", "Malayalam": "ml-IN", "Hindi": "hi-IN", "Tamil": "ta-IN"}


def plain_summary(prediction: dict) -> str:
    """'The model predicts Rejected, with a 3% chance of approval. ...'"""
    p = prediction
    factors = "; ".join(f"{f['name']} of {format_value(f['feature'], f['value']).replace('Rs ', 'rupees ')}, "
                        f"which pushes {f['direction']}" for f in p["factors"])
    return (f"The model predicts {p['label']}, with a {p['probability']:.0%} chance of approval. "
            f"This is a {p['band']} result. The main factors are: {factors}. "
            "This is an educational estimate, not a lending decision.")


def strip_for_speech(text: str) -> str:
    """Remove things that sound bad when read aloud."""
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.M)
    return re.sub(r"[ \t]+", " ", text).strip()


def speak_html(text: str, lang: str = "en-IN") -> str:
    """Self-contained Play / Stop buttons that read `text` with a matching browser voice."""
    payload = json.dumps(strip_for_speech(text))
    return f"""
<div style="font-family:system-ui,sans-serif;display:flex;gap:.5rem;align-items:center;flex-wrap:wrap">
  <button id="play" style="padding:.45rem 1rem;border-radius:8px;border:1px solid #C7D2FE;background:#EEF2FF;
          color:#3730A3;font-weight:600;cursor:pointer">▶ Listen</button>
  <button id="stop" style="padding:.45rem 1rem;border-radius:8px;border:1px solid #E5E7EB;background:white;
          color:#374151;cursor:pointer">■ Stop</button>
  <span id="msg" style="font-size:.82rem;color:#6B7280"></span>
</div>
<script>
const text = {payload}, lang = {json.dumps(lang)}, msg = document.getElementById("msg");
function voiceFor(l) {{
  const vs = speechSynthesis.getVoices();
  return vs.find(v => v.lang === l) || vs.find(v => v.lang.startsWith(l.split("-")[0]));
}}
document.getElementById("play").onclick = () => {{
  if (!("speechSynthesis" in window)) {{ msg.textContent = "This browser cannot read text aloud."; return; }}
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text), v = voiceFor(lang);
  if (!v && !lang.startsWith("en")) {{
    msg.textContent = "No " + lang + " voice is installed on this device; add one in your OS speech settings.";
    return;
  }}
  if (v) u.voice = v;
  u.lang = lang; u.rate = 0.95;
  u.onend = () => msg.textContent = "";
  msg.textContent = "Reading" + (v ? " with " + v.name : "") + "…";
  speechSynthesis.speak(u);
}};
document.getElementById("stop").onclick = () => {{ speechSynthesis.cancel(); msg.textContent = ""; }};
</script>"""


def escape_for_card(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")
