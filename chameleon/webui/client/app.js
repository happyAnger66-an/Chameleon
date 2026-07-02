"use strict";

// Chameleon WebUI 前端 (M1 配置 + M2 运行/日志)。
// 与后端约定的接口：
//   GET  /api/configs             -> { configs, configs_dir, commands }
//   GET  /api/configs/{name}      -> { name, text, actions }
//   POST /api/validate {text}     -> { ok, error, actions }
//   POST /api/run {command,text,interpreter,config_name} -> run snapshot
//   POST /api/runs/{id}/cancel    -> { cancelled }
//   WS   /ws/run/{id}             -> {type:log,line} | {type:status,...}

const $ = (id) => document.getElementById(id);

const el = {
  configSelect: $("configSelect"),
  commandSelect: $("commandSelect"),
  interpreter: $("interpreter"),
  btnReload: $("btnReload"),
  btnValidate: $("btnValidate"),
  btnRun: $("btnRun"),
  btnCancel: $("btnCancel"),
  status: $("status"),
  editor: $("editor"),
  editorMeta: $("editorMeta"),
  actionsList: $("actionsList"),
  validateResult: $("validateResult"),
  logConsole: $("logConsole"),
  runMeta: $("runMeta"),
};

let currentName = "";
let ws = null;
let currentRunId = "";

function setStatus(text, kind = "gray") {
  el.status.textContent = text;
  el.status.className = `badge badge-${kind}`;
}

function renderActions(actions) {
  el.actionsList.innerHTML = "";
  if (!actions || actions.length === 0) {
    const li = document.createElement("li");
    li.className = "muted";
    li.textContent = "该配置未声明 actions";
    el.actionsList.appendChild(li);
    return;
  }
  for (const a of actions) {
    const li = document.createElement("li");
    li.textContent = a;
    el.actionsList.appendChild(li);
  }
}

function setValidate(text, kind = "muted") {
  el.validateResult.className = `validateResult ${kind}`;
  el.validateResult.textContent = text;
}

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch (_) {
      /* ignore */
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json();
}

function populateCommands(commands) {
  const list = commands && commands.length ? commands : ["workflow"];
  const prev = el.commandSelect.value || "workflow";
  el.commandSelect.innerHTML = "";
  for (const c of list) {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    el.commandSelect.appendChild(opt);
  }
  el.commandSelect.value = list.includes(prev) ? prev : list[0];
}

async function loadConfigList() {
  setStatus("加载配置列表…", "busy");
  try {
    const data = await api("/api/configs");
    const names = data.configs || [];
    el.configSelect.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = names.length
      ? `— 选择配置（共 ${names.length}）—`
      : "（configs 目录下无 YAML）";
    el.configSelect.appendChild(placeholder);
    for (const name of names) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      el.configSelect.appendChild(opt);
    }
    populateCommands(data.commands);
    el.editorMeta.textContent = data.configs_dir || "";
    setStatus("就绪", "gray");
  } catch (err) {
    setStatus("列表加载失败", "err");
    setValidate(String(err), "err");
  }
}

async function loadConfig(name) {
  if (!name) {
    currentName = "";
    el.editor.value = "";
    renderActions([]);
    setValidate("尚未校验", "muted");
    return;
  }
  setStatus(`读取 ${name}…`, "busy");
  try {
    const data = await api(`/api/configs/${encodeURIComponent(name)}`);
    currentName = data.name;
    el.editor.value = data.text || "";
    renderActions(data.actions);
    setValidate("已载入，尚未校验", "muted");
    setStatus("就绪", "gray");
  } catch (err) {
    setStatus("读取失败", "err");
    setValidate(String(err), "err");
  }
}

async function validateCurrent() {
  setStatus("校验中…", "busy");
  try {
    const data = await api("/api/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: el.editor.value }),
    });
    renderActions(data.actions);
    if (data.ok) {
      setValidate("✓ 配置有效（schema 校验通过）", "ok");
      setStatus("校验通过", "ok");
    } else {
      setValidate(`✗ 校验失败:\n${data.error || "未知错误"}`, "err");
      setStatus("校验失败", "err");
    }
  } catch (err) {
    setValidate(String(err), "err");
    setStatus("校验出错", "err");
  }
}

function appendLog(line) {
  const atBottom =
    el.logConsole.scrollTop + el.logConsole.clientHeight >=
    el.logConsole.scrollHeight - 4;
  el.logConsole.textContent += (el.logConsole.textContent ? "\n" : "") + line;
  if (atBottom) el.logConsole.scrollTop = el.logConsole.scrollHeight;
}

function setRunning(running) {
  el.btnRun.disabled = running;
  el.btnCancel.disabled = !running;
  el.configSelect.disabled = running;
  el.commandSelect.disabled = running;
  el.interpreter.disabled = running;
}

function statusKind(status) {
  if (status === "done") return "ok";
  if (status === "failed" || status === "cancelled") return "err";
  if (status === "running") return "busy";
  return "gray";
}

function openRunSocket(runId) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${location.host}/ws/run/${runId}`;
  ws = new WebSocket(url);
  ws.onmessage = (ev) => {
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch (_) {
      return;
    }
    if (msg.type === "log") {
      appendLog(msg.line);
    } else if (msg.type === "status") {
      el.runMeta.textContent = `${msg.command} · ${msg.run_id} · ${msg.status}`;
      setStatus(msg.status, statusKind(msg.status));
      if (msg.status !== "running") {
        setRunning(false);
        if (msg.returncode !== null && msg.returncode !== undefined) {
          appendLog(`[ui] 结束状态=${msg.status} returncode=${msg.returncode}`);
        }
      }
    } else if (msg.type === "error") {
      appendLog(`[ui] ${msg.message}`);
      setStatus("错误", "err");
      setRunning(false);
    }
  };
  ws.onclose = () => {
    ws = null;
  };
  ws.onerror = () => {
    appendLog("[ui] WebSocket 连接错误");
  };
}

async function runCurrent() {
  const command = el.commandSelect.value || "workflow";
  el.logConsole.textContent = "";
  setStatus("启动中…", "busy");
  setRunning(true);
  try {
    const snap = await api("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        command,
        text: el.editor.value,
        interpreter: el.interpreter.value || null,
        config_name: currentName || null,
      }),
    });
    currentRunId = snap.run_id;
    el.runMeta.textContent = `${snap.command} · ${snap.run_id} · ${snap.status}`;
    setStatus("运行中", "busy");
    openRunSocket(currentRunId);
  } catch (err) {
    appendLog(`[ui] 启动失败: ${err}`);
    setStatus("启动失败", "err");
    setRunning(false);
  }
}

async function cancelCurrent() {
  if (!currentRunId) return;
  el.btnCancel.disabled = true;
  try {
    await api(`/api/runs/${currentRunId}/cancel`, { method: "POST" });
    appendLog("[ui] 已请求取消…");
  } catch (err) {
    appendLog(`[ui] 取消失败: ${err}`);
    el.btnCancel.disabled = false;
  }
}

el.configSelect.addEventListener("change", (e) => loadConfig(e.target.value));
el.btnReload.addEventListener("click", () => loadConfig(currentName));
el.btnValidate.addEventListener("click", validateCurrent);
el.btnRun.addEventListener("click", runCurrent);
el.btnCancel.addEventListener("click", cancelCurrent);

loadConfigList();
