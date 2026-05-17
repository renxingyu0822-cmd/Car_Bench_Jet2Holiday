"""Web chat UI for interacting with the purple agent as the green agent."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).parent))
from agentbeats.client import send_message_with_parts
from a2a.types import DataPart, Part, TextPart

app = FastAPI(title="Purple Agent Chat")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# session_id -> {context_id, task_id, turn}
sessions: dict = {}

# Cached CAR-bench config
_carbench_cache: dict | None = None


def _load_carbench_config() -> dict:
    global _carbench_cache
    if _carbench_cache is not None:
        return _carbench_cache

    repo = Path(__file__).parent.parent / "scenarios" / "car-bench" / "car-bench"
    import os, argparse
    sys.path.insert(0, str(repo))
    os.environ.setdefault(
        "CAR_BENCH_DATA_DIR",
        str(repo / "car_bench" / "envs" / "car_voice_assistant" / "mock_data"),
    )

    captured: dict = {}

    def capture_factory(tools_info, wiki, args):
        captured["tools"] = tools_info
        captured["system_prompt"] = wiki  # wiki is the system prompt string

        class _Dummy:
            def get_init_state(self, sp, obs):
                captured["system_prompt"] = sp
                import types
                s = types.SimpleNamespace(
                    messages=[], total_cost=0,
                    total_llm_induced_latency_ms=0, turn_counter=0,
                    least_prompt_tokens=0, latest_prompt_tokens=0,
                )
                return s

            def generate_next_message(self, state, tools_info):
                msg = {"role": "assistant", "content": "###STOP###", "tool_calls": None}
                return msg, state

        return _Dummy()

    from run import run as _run
    args = argparse.Namespace(
        env="car_voice_assistant", task_type="base", task_split="test",
        num_tasks=1, task_id_filter=None, num_trials=1, max_concurrency=1,
        user_strategy="llm", user_model="openai/gpt-4o-mini", user_model_provider="openai",
        user_thinking=False, policy_evaluator_strategy="llm",
        policy_evaluator_model="openai/gpt-4o-mini", policy_evaluator_model_provider="openai",
        evaluate_policy=False, score_tool_execution_errors=False, score_policy_errors=False,
        agent_strategy="tool-calling", model="dummy", model_provider="dummy",
        temperature=0.0, thinking=False, interleaved_thinking=False, reasoning_effort="none",
        use_user_as_a_tool_tools=False, planning_and_thinking_tool=True,
        remove_non_standard_fields_from_tools=False, few_shot_displays_path=None,
        seed=10, shuffle=False, max_steps=1,
    )
    try:
        _run(args, "/tmp/_chat_server_probe.json", capture_factory)
    except Exception:
        pass
    finally:
        sys.path.pop(0)

    _carbench_cache = captured
    return captured


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.get("/api/carbench-config")
async def carbench_config():
    try:
        cfg = await asyncio.get_event_loop().run_in_executor(None, _load_carbench_config)
        return JSONResponse({
            "ok": True,
            "system_prompt": cfg.get("system_prompt", ""),
            "tools": cfg.get("tools", []),
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/reset")
async def reset_session(request: Request):
    data = await request.json()
    sid = data.get("session_id", "default")
    sessions[sid] = {"context_id": None, "task_id": None, "turn": 0}
    return {"ok": True}


@app.post("/api/send")
async def send(request: Request):
    data = await request.json()
    sid = data.get("session_id", "default")
    agent_url = data.get("agent_url", "http://127.0.0.1:8080")

    if sid not in sessions:
        sessions[sid] = {"context_id": None, "task_id": None, "turn": 0}

    sess = sessions[sid]
    msg_type = data.get("type", "user")  # "first" | "user" | "tool_results"

    parts: list[Part] = []

    if msg_type == "first":
        sys_prompt = data.get("system_prompt", "")
        user_text = data.get("text", "")
        tools = data.get("tools", [])
        combined = f"System: {sys_prompt}\n\nUser: {user_text}" if sys_prompt else user_text
        parts.append(Part(root=TextPart(kind="text", text=combined)))
        if tools:
            parts.append(Part(root=DataPart(kind="data", data={"tools": tools})))

    elif msg_type == "tool_results":
        tool_results = data.get("tool_results", [])
        parts.append(Part(root=DataPart(kind="data", data={"tool_results": tool_results})))

    else:
        text = data.get("text", "")
        parts.append(Part(root=TextPart(kind="text", text=text)))

    try:
        new_conv = sess["turn"] == 0
        outputs = await send_message_with_parts(
            parts=parts,
            base_url=agent_url,
            context_id=None if new_conv else sess["context_id"],
            task_id=None if new_conv else sess["task_id"],
        )

        if outputs.get("status", "completed") != "completed":
            return JSONResponse(
                {"ok": False, "error": f"Agent returned status: {outputs.get('status')}"},
                status_code=500,
            )

        sess["context_id"] = outputs.get("context_id")
        sess["task_id"] = outputs.get("task_id")
        sess["turn"] += 1

        raw_msg = outputs.get("raw_message")
        response_text = None
        tool_calls = None

        if raw_msg and hasattr(raw_msg, "parts"):
            for part in raw_msg.parts:
                if not hasattr(part, "root"):
                    continue
                root = part.root
                if isinstance(root, TextPart):
                    response_text = root.text
                elif isinstance(root, DataPart):
                    d = root.data
                    if "tool_calls" in d:
                        tool_calls = d["tool_calls"]

        return JSONResponse({
            "ok": True,
            "text": response_text,
            "tool_calls": tool_calls,
            "context_id": sess["context_id"],
            "task_id": sess["task_id"],
            "turn": sess["turn"],
        })

    except Exception as e:
        import traceback
        return JSONResponse(
            {"ok": False, "error": str(e), "detail": traceback.format_exc()},
            status_code=500,
        )


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Purple Agent Chat</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f0f1a;color:#e0e0e0;height:100vh;display:flex;overflow:hidden}

/* ── Sidebar ── */
#sidebar{width:300px;min-width:300px;background:#141428;border-right:1px solid #222244;display:flex;flex-direction:column;padding:16px;gap:14px;overflow-y:auto}
#sidebar h2{color:#a78bfa;font-size:12px;text-transform:uppercase;letter-spacing:1.5px;padding-bottom:6px;border-bottom:1px solid #222244}
.cfg-group{display:flex;flex-direction:column;gap:5px}
.cfg-group label{font-size:11px;color:#666;text-transform:uppercase;letter-spacing:0.5px}
.cfg-group input,.cfg-group textarea{background:#0f0f1a;border:1px solid #222244;border-radius:7px;color:#e0e0e0;padding:8px 10px;font-size:12px;font-family:inherit;outline:none;transition:border-color .2s;resize:vertical}
.cfg-group input:focus,.cfg-group textarea:focus{border-color:#a78bfa}
.btn{padding:8px 14px;border-radius:7px;border:none;cursor:pointer;font-size:12px;font-weight:600;transition:opacity .15s}
.btn:hover{opacity:.82}.btn:disabled{opacity:.35;cursor:not-allowed}
.btn-purple{background:#7c3aed;color:#fff}
.btn-red{background:#be185d;color:#fff}
.btn-gray{background:#1e1e38;color:#aaa}
.btn-sm{padding:4px 10px;font-size:11px}
#status{padding:6px 10px;border-radius:6px;font-size:11px;background:#1e1e38;color:#666;transition:all .2s}
#status.ok{background:#14302a;color:#4ade80}
#status.error{background:#3b0a0a;color:#f87171}
#status.busy{background:#1e1e38;color:#fb923c}
.hint{font-size:10px;color:#383858;line-height:1.7;margin-top:auto}
.hint b{color:#444466}

/* ── Main ── */
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#chat-header{background:#141428;border-bottom:1px solid #222244;padding:12px 20px;display:flex;align-items:center;gap:10px;flex-shrink:0}
#chat-header h1{font-size:14px;font-weight:700}
.badge{background:#1e1e38;border-radius:20px;padding:2px 10px;font-size:11px;color:#666}

/* ── Messages ── */
#messages{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:14px}
.msg{max-width:78%;display:flex;flex-direction:column;gap:5px}
.msg.user{align-self:flex-end;align-items:flex-end}
.msg.agent{align-self:flex-start;align-items:flex-start}
.msg.sys{align-self:center;align-items:center}
.msg-label{font-size:10px;color:#444;text-transform:uppercase;letter-spacing:.5px}
.bubble{padding:10px 14px;border-radius:12px;font-size:13px;line-height:1.6;white-space:pre-wrap;word-break:break-word}
.msg.user .bubble{background:#3b0764;border-bottom-right-radius:3px}
.msg.agent .bubble{background:#1e1e38;border-bottom-left-radius:3px}
.msg.sys .bubble{background:#141428;color:#555;font-size:11px;border-radius:20px;padding:5px 14px}

/* Tool call card */
.tc-card{background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;overflow:hidden;font-size:12px;width:100%}
.tc-header{background:#1e3a5f;padding:6px 12px;display:flex;align-items:center;gap:7px;font-weight:700;color:#93c5fd}
.tc-body{padding:10px 12px;font-family:'SF Mono','Fira Code',monospace;color:#7dd3fc;overflow-x:auto;white-space:pre}

/* Tool result card */
.tr-card{background:#0a2018;border:1px solid #14532d;border-radius:10px;overflow:hidden;font-size:12px;width:100%}
.tr-header{background:#14532d;padding:6px 12px;font-weight:700;color:#86efac}
.tr-body{padding:10px 12px;font-family:'SF Mono','Fira Code',monospace;color:#4ade80;overflow-x:auto;white-space:pre-wrap;word-break:break-all}

/* Loading */
.loading-row{display:flex;align-items:center;gap:8px;color:#555;font-size:12px}
.spinner{width:14px;height:14px;border:2px solid #222244;border-top-color:#a78bfa;border-radius:50%;animation:spin .7s linear infinite;flex-shrink:0}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── Tool result input area ── */
#tr-area{background:#141428;border-top:1px solid #222244;padding:14px 16px;display:none;flex-direction:column;gap:10px;flex-shrink:0;max-height:45vh;overflow-y:auto}
#tr-area.show{display:flex}
#tr-area h3{font-size:11px;color:#a78bfa;text-transform:uppercase;letter-spacing:1px}
.tr-input-card{background:#0f0f1a;border:1px solid #222244;border-radius:8px;padding:10px;display:flex;flex-direction:column;gap:7px}
.tr-input-card .tname{font-size:11px;color:#60a5fa;font-weight:700;font-family:'SF Mono','Fira Code',monospace}
.tr-input-card .targs{font-size:10px;color:#374151;background:#0a0a14;border-radius:5px;padding:5px 8px;font-family:'SF Mono','Fira Code',monospace;white-space:pre;overflow-x:auto;max-height:80px}
.tr-input-card textarea{background:#0a0a14;border:1px solid #222244;border-radius:6px;color:#e0e0e0;padding:7px 9px;font-size:12px;font-family:'SF Mono','Fira Code',monospace;resize:vertical;min-height:52px;outline:none}
.tr-input-card textarea:focus{border-color:#a78bfa}
#tr-send-row{display:flex;gap:8px;align-items:center}

/* ── Input area ── */
#input-area{background:#141428;border-top:1px solid #222244;padding:12px 16px;display:flex;gap:10px;align-items:flex-end;flex-shrink:0}
#user-input{flex:1;background:#0f0f1a;border:1px solid #222244;border-radius:10px;color:#e0e0e0;padding:9px 13px;font-size:13px;font-family:inherit;resize:none;outline:none;min-height:40px;max-height:140px;transition:border-color .2s;line-height:1.45}
#user-input:focus{border-color:#a78bfa}
#send-btn{width:40px;height:40px;border-radius:50%;border:none;background:#7c3aed;color:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:17px;transition:opacity .15s,transform .1s;flex-shrink:0}
#send-btn:hover{opacity:.82}
#send-btn:active{transform:scale(.93)}
#send-btn:disabled{opacity:.3;cursor:not-allowed}

::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#222244;border-radius:3px}
</style>
</head>
<body>

<div id="sidebar">
  <h2>⚙ Config</h2>

  <div class="cfg-group">
    <label>Purple Agent URL</label>
    <input id="agent-url" value="http://127.0.0.1:8080">
  </div>

  <button class="btn btn-purple" id="load-btn" onclick="loadCarbench()" style="width:100%">
    🚗 Load CAR-bench Config
  </button>
  <div id="load-status" style="font-size:11px;color:#555;display:none"></div>

  <div class="cfg-group">
    <label>System Prompt <span style="color:#333">(first message only)</span></label>
    <textarea id="system-prompt" rows="4" placeholder="Click 'Load CAR-bench Config' above, or paste manually"></textarea>
  </div>

  <div class="cfg-group">
    <label>Tools JSON <span style="color:#333">(57 tools after loading)</span></label>
    <textarea id="tools-json" rows="4" placeholder="Click 'Load CAR-bench Config' above, or paste manually"></textarea>
    <button class="btn btn-gray btn-sm" onclick="clearTools()">Clear</button>
  </div>

  <div id="status">Ready</div>
  <button class="btn btn-red" onclick="resetConversation()">↺ Reset Conversation</button>

  <div class="hint">
    <b>Protocol flow:</b><br>
    Turn 1 → TextPart(system+user) + DataPart(tools)<br>
    Tool results → DataPart({tool_results})<br>
    Next turns → TextPart(user message)
  </div>
</div>

<div id="main">
  <div id="chat-header">
    <span style="font-size:18px">🟣</span>
    <h1>Purple Agent Chat</h1>
    <span class="badge" id="turn-badge">Turn 0</span>
    <span class="badge" id="ctx-badge" style="margin-left:auto;display:none"></span>
  </div>

  <div id="messages">
    <div class="msg sys"><div class="bubble">Configure the agent URL, then type your first message.</div></div>
  </div>

  <!-- Tool result inputs -->
  <div id="tr-area">
    <h3>🔧 Enter Tool Results</h3>
    <div id="tr-inputs"></div>
    <div id="tr-send-row">
      <button class="btn btn-purple" id="tr-send-btn" onclick="sendToolResults()">Send Results →</button>
      <span style="font-size:11px;color:#555">Fill in each result, then click Send Results</span>
    </div>
  </div>

  <!-- Regular text input -->
  <div id="input-area">
    <textarea id="user-input" placeholder="Type your message… (Enter to send, Shift+Enter for newline)"
      onkeydown="onKey(event)" oninput="grow(this)"></textarea>
    <button id="send-btn" onclick="sendMessage()" title="Send">➤</button>
  </div>
</div>

<script>
const SID = Math.random().toString(36).slice(2);
let isFirst = true, loading = false, pendingTC = [];

// ── Utilities ──────────────────────────────────────────
const esc = s => (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const $  = id => document.getElementById(id);
const msgs = () => $('messages');

function setStatus(txt, cls=''){
  $('status').textContent = txt;
  $('status').className = cls;
}

function scrollBottom(){
  const m = msgs();
  m.scrollTop = m.scrollHeight;
}

function grow(el){
  el.style.height='auto';
  el.style.height = Math.min(el.scrollHeight,140)+'px';
}

function setBusy(b){
  loading = b;
  $('send-btn').disabled = b;
  $('user-input').disabled = b;
  if($('tr-send-btn')) $('tr-send-btn').disabled = b;
}

// ── Message rendering ───────────────────────────────────
function addSys(txt){
  msgs().insertAdjacentHTML('beforeend',
    `<div class="msg sys"><div class="bubble">${esc(txt)}</div></div>`);
  scrollBottom();
}

function addUser(txt){
  msgs().insertAdjacentHTML('beforeend',`
    <div class="msg user">
      <div class="msg-label">You</div>
      <div class="bubble">${esc(txt)}</div>
    </div>`);
  scrollBottom();
}

function addAgent(text, toolCalls){
  let html = '<div class="msg agent"><div class="msg-label">Purple Agent</div>';
  if(text) html += `<div class="bubble">${esc(text)}</div>`;
  if(toolCalls?.length){
    for(const tc of toolCalls){
      const args = JSON.stringify(tc.arguments||{}, null, 2);
      html += `
        <div class="tc-card">
          <div class="tc-header">🔧 ${esc(tc.tool_name)}</div>
          <div class="tc-body">${esc(args)}</div>
        </div>`;
    }
  }
  html += '</div>';
  msgs().insertAdjacentHTML('beforeend', html);
  scrollBottom();
}

function addToolResultMsg(results){
  let html = '<div class="msg user"><div class="msg-label">Tool Results (You)</div>';
  for(const r of results){
    html += `
      <div class="tr-card">
        <div class="tr-header">✓ ${esc(r.tool_name)}</div>
        <div class="tr-body">${esc(r.content)}</div>
      </div>`;
  }
  html += '</div>';
  msgs().insertAdjacentHTML('beforeend', html);
  scrollBottom();
}

function addLoading(){
  msgs().insertAdjacentHTML('beforeend',`
    <div class="msg agent" id="loading-msg">
      <div class="loading-row"><div class="spinner"></div><span>Thinking…</span></div>
    </div>`);
  scrollBottom();
}

function removeLoading(){ $('loading-msg')?.remove(); }

// ── Tool result area ────────────────────────────────────
function showTRArea(toolCalls){
  pendingTC = toolCalls;
  const inputs = $('tr-inputs');
  inputs.innerHTML = '';
  for(const tc of toolCalls){
    const args = JSON.stringify(tc.arguments||{}, null, 2);
    const div = document.createElement('div');
    div.className = 'tr-input-card';
    div.dataset.toolName = tc.tool_name;
    div.innerHTML = `
      <div class="tname">🔧 ${esc(tc.tool_name)}</div>
      <div class="targs">${esc(args)}</div>
      <textarea placeholder="Enter result for ${esc(tc.tool_name)}…" rows="2"></textarea>`;
    inputs.appendChild(div);
  }
  $('tr-area').classList.add('show');
  $('input-area').style.display = 'none';
  inputs.querySelector('textarea')?.focus();
}

function hideTRArea(){
  $('tr-area').classList.remove('show');
  $('input-area').style.display = 'flex';
  pendingTC = [];
}

// ── API calls ───────────────────────────────────────────
async function post(path, body){
  const r = await fetch(path, {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)
  });
  return r.json();
}

function handleResponse(result){
  removeLoading();
  if(!result.ok){
    setStatus('Error: '+(result.error||'unknown'), 'error');
    addSys('Error: '+(result.error||'unknown'));
    setBusy(false);
    return false;
  }
  isFirst = false;
  $('turn-badge').textContent = `Turn ${result.turn}`;
  if(result.context_id){
    const cb = $('ctx-badge');
    cb.style.display='';
    cb.textContent = 'ctx: '+result.context_id.slice(0,8);
  }
  addAgent(result.text, result.tool_calls);
  if(result.tool_calls?.length){
    showTRArea(result.tool_calls);
    setStatus(`${result.tool_calls.length} tool call(s) — enter results below`, 'busy');
  } else {
    setBusy(false);
    setStatus('Ready', 'ok');
    $('user-input').focus();
  }
  return true;
}

async function sendMessage(){
  if(loading) return;
  const text = $('user-input').value.trim();
  if(!text) return;

  const agentUrl = $('agent-url').value.trim();
  let tools = [];
  const tj = $('tools-json').value.trim();
  if(tj){
    try{ tools = JSON.parse(tj); }
    catch(e){ setStatus('Invalid tools JSON: '+e.message,'error'); return; }
  }

  addUser(text);
  $('user-input').value = '';
  $('user-input').style.height = 'auto';
  setBusy(true);
  setStatus('Sending…', 'busy');
  addLoading();

  const payload = isFirst
    ? {session_id:SID, agent_url:agentUrl, type:'first',
       system_prompt: $('system-prompt').value.trim(), text, tools}
    : {session_id:SID, agent_url:agentUrl, type:'user', text};

  try{
    const result = await post('/api/send', payload);
    handleResponse(result);
  } catch(e){
    removeLoading();
    setStatus('Network error: '+e.message, 'error');
    addSys('Network error: '+e.message);
    setBusy(false);
  }
}

async function sendToolResults(){
  if(loading) return;
  const cards = $('tr-inputs').querySelectorAll('.tr-input-card');
  const toolResults = [];
  for(const card of cards){
    toolResults.push({
      tool_name: card.dataset.toolName,
      tool_call_id: card.dataset.toolName,
      content: card.querySelector('textarea').value
    });
  }

  addToolResultMsg(toolResults);
  hideTRArea();
  setBusy(true);
  setStatus('Sending tool results…', 'busy');
  addLoading();

  const agentUrl = $('agent-url').value.trim();
  try{
    const result = await post('/api/send',
      {session_id:SID, agent_url:agentUrl, type:'tool_results', tool_results:toolResults});
    handleResponse(result);
  } catch(e){
    removeLoading();
    setStatus('Network error: '+e.message,'error');
    addSys('Network error: '+e.message);
    setBusy(false);
  }
}

async function resetConversation(){
  await post('/api/reset', {session_id:SID});
  isFirst = true; pendingTC = [];
  hideTRArea(); setBusy(false);
  msgs().innerHTML = '';
  $('turn-badge').textContent = 'Turn 0';
  $('ctx-badge').style.display = 'none';
  addSys('Conversation reset. Type your first message.');
  setStatus('Ready');
}

function clearTools(){ $('tools-json').value=''; }

async function loadCarbench(){
  const btn = $('load-btn');
  const st = $('load-status');
  btn.disabled = true;
  btn.textContent = '⏳ Loading…';
  st.style.display = '';
  st.textContent = 'Fetching tools and system prompt from CAR-bench (first load may take ~30s)…';
  try {
    const r = await fetch('/api/carbench-config');
    const d = await r.json();
    if(!d.ok){ st.textContent = 'Error: '+d.error; btn.disabled=false; btn.textContent='🚗 Load CAR-bench Config'; return; }
    $('system-prompt').value = d.system_prompt || '';
    $('tools-json').value = JSON.stringify(d.tools, null, 2);
    st.textContent = `✓ Loaded ${d.tools.length} tools + system prompt`;
    st.style.color = '#4ade80';
    btn.textContent = '✓ Loaded';
  } catch(e) {
    st.textContent = 'Network error: '+e.message;
    btn.disabled = false;
    btn.textContent = '🚗 Load CAR-bench Config';
  }
}

function onKey(e){
  if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); sendMessage(); }
}
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Purple Agent Chat Web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    print(f"\n  Chat UI → http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
