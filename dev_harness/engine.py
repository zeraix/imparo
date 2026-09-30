#!/usr/bin/env python3
"""Engine-independent driver: launch a server, drive it over the OpenAI-compatible API, sample RSS.

Deliberately measures wall-clock and OpenAI `usage` counts only, never an engine's own timing
report, so the fork and Imparo are measured by the same ruler.
"""
import json, os, re, signal, subprocess, time, urllib.request, urllib.error, threading


def wired_mib():
    """System-wide wired (kernel-locked) memory.

    On Apple Silicon a Metal buffer created over an mmap with StorageModeShared can be
    wired, and wired pages are NOT charged to process RSS. Comparing engines on RSS alone
    therefore flatters whichever one avoids copying weights. This is the system-level
    number that has to be reported next to the per-process one.
    """
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    except Exception:
        return 0.0
    page = 4096
    m = re.search(r"page size of (\d+) bytes", out)
    if m:
        page = int(m.group(1))
    m = re.search(r"Pages wired down:\s+(\d+)", out)
    return (int(m.group(1)) * page / (1 << 20)) if m else 0.0


def phys_footprint_mib(pid):
    """macOS phys_footprint: what the OS actually charges the process."""
    try:
        out = subprocess.run(["footprint", "-p", str(pid)], capture_output=True,
                             text=True, timeout=20).stdout
    except Exception:
        return 0.0
    m = re.search(r"([\d.]+)\s*([KMG])B\s+phys_footprint", out)
    if not m:
        m = re.search(r"phys_footprint:\s*([\d.]+)\s*([KMG])", out)
    if not m:
        return 0.0
    v = float(m.group(1))
    return v * {"K": 1 / 1024, "M": 1.0, "G": 1024.0}[m.group(2)]

class RssSampler(threading.Thread):
    """Peak RSS of a process tree, sampled. Peak matters more than final: the load spike is real."""
    def __init__(self, pid, interval=0.25):
        super().__init__(daemon=True)
        self.pid, self.interval, self.peak_kb, self.stop_flag = pid, interval, 0, False
        self.peak_wired_mib = 0.0
        self.base_wired_mib = wired_mib()
    def _tree_rss_kb(self):
        if os.name == "nt":
            try:
                out = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {self.pid}", "/FO", "CSV", "/NH"],
                    capture_output=True, text=True, timeout=5).stdout
                match = re.search(r'"([\d,.]+)\s+K"\s*$', out.strip())
                return int(match.group(1).replace(",", "").replace(".", "")) \
                    if match else 0
            except Exception:
                return 0
        try:
            out = subprocess.run(["ps", "-eo", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
        except Exception:
            return 0
        kids, rss = {}, {}
        for line in out.splitlines():
            f = line.split()
            if len(f) != 3: continue
            pid, ppid, r = int(f[0]), int(f[1]), int(f[2])
            kids.setdefault(ppid, []).append(pid); rss[pid] = r
        total, stack = 0, [self.pid]
        while stack:
            p = stack.pop()
            total += rss.get(p, 0)
            stack.extend(kids.get(p, []))
        return total
    def run(self):
        while not self.stop_flag:
            self.peak_kb = max(self.peak_kb, self._tree_rss_kb())
            self.peak_wired_mib = max(self.peak_wired_mib, wired_mib())
            time.sleep(self.interval)
    def stop(self):
        self.stop_flag = True; self.join(timeout=2); return self.peak_kb


class Server:
    """One OpenAI-compatible server, launched and driven here.

    `api_key` goes out as a bearer token (oMLX refuses every request without one), `model`
    is the name the requests carry ("local" for imparo and llama-server, the model
    directory's name for oMLX), and `process_group` starts the command in its own session
    so a CLI that forks the real server (omlx serve -> omlx-server) is stopped with its
    child. `cmd=None` describes a server this harness did not start and must not stop,
    reached at `base_url`.
    """
    def __init__(self, cmd, port, log_path, name, api_key=None, model="local",
                 process_group=False, base_url=None):
        self.cmd, self.port, self.log_path, self.name = cmd, port, log_path, name
        self.api_key, self.model, self.process_group = api_key, model, process_group
        self.base_url = base_url or f"http://127.0.0.1:{port}"
        self.proc = self.log = self.rss = None
        self.start_seconds = None
        self.health_seconds = None

    def kill_strays(self):
        """An orphan still answers /health while the new bind fails -> you measure the old server."""
        if os.name == "nt":
            out = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True,
                                 text=True).stdout
            pids = set()
            for line in out.splitlines():
                fields = line.split()
                if len(fields) >= 5 and fields[1].endswith(f":{self.port}") \
                        and fields[3].upper() == "LISTENING":
                    pids.add(fields[4])
            for pid in pids:
                subprocess.run(["taskkill", "/PID", pid, "/T", "/F"],
                               capture_output=True)
        else:
            for pat in ("llama-server", "imparo-server", "zeraix", "omlx", "rapid-mlx"):
                subprocess.run(["pkill", "-f", f"{pat}.*--port {self.port}"],
                               capture_output=True)
        for _ in range(40):
            if not self._port_answers(): return True
            time.sleep(0.25)
        return False

    def _port_answers(self):
        """Something is listening: a 200, or any HTTP error (oMLX has no /health and answers
        404; llama-server answers 503 while loading). Service is proven by start()'s
        completion, not here."""
        try:
            urllib.request.urlopen(f"{self.base_url}/health", timeout=1); return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            return False

    def start(self, ready_timeout=900):
        """start_seconds is time until the server can SERVE, proven by a real completion.

        /health answers 200 while the model is still loading -- the first attempt at this
        measured 1.03s against a log that showed the model still initialising at 0.99s. A
        readiness signal that does not prove service is not a start time.
        """
        if not self.kill_strays():
            raise RuntimeError(f"port {self.port} still answering after pkill; refusing to measure")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.log = open(self.log_path, "w")
        t0 = time.time()
        env = getattr(self, "cmd_env", None)
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     env=env, creationflags=creationflags,
                                     start_new_session=self.process_group and os.name != "nt")
        self.rss = RssSampler(self.proc.pid); self.rss.start()
        deadline = t0 + ready_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.name} exited during startup, rc={self.proc.returncode}; see {self.log_path}")
            if self._port_answers():
                if self.health_seconds is None:
                    self.health_seconds = time.time() - t0
                try:
                    self.chat([{"role": "user", "content": "hi"}], "readiness-probe", max_tokens=1)
                    self.start_seconds = time.time() - t0
                    return self.start_seconds
                except Exception:
                    pass
            time.sleep(0.2)
        raise RuntimeError(f"{self.name} not ready in {ready_timeout}s; see {self.log_path}")

    def chat(self, messages, conv_id, max_tokens=128, temperature=0.0, seed=1234, tools=None,
             template_kwargs=None):
        """One streaming /v1/chat/completions call. Returns per-stage wall timings + usage.

        `template_kwargs` is the request's `chat_template_kwargs` (llama.cpp's field, taken
        by imparo, llama-server, oMLX and rapid-mlx), e.g. {"enable_thinking": False}.
        `last_s` is the arrival of the last generated chunk and `chunks` the number of
        them, which is what a client-side decode rate is made of; `chunk_times` is when
        each arrived, seconds from the request; `timings` is the
        server's own block when it sends one (imparo and llama-server do, in the final
        frame), None otherwise.
        """
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "seed": seed, "stream": True,
                "stream_options": {"include_usage": True}}
        if tools: body["tools"] = tools
        if template_kwargs is not None: body["chat_template_kwargs"] = template_kwargs
        headers = {"Content-Type": "application/json", "X-Conversation-Id": conv_id}
        if self.api_key: headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(body).encode(), method="POST", headers=headers)
        t0 = time.time(); ttft = None; last = None; text = []; reasoning = []
        tool_calls = {}; usage = None; timings = None; n_chunks = 0; chunk_times = []
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "): continue
                payload = line[6:]
                if payload == "[DONE]": break
                try: obj = json.loads(payload)
                except json.JSONDecodeError: continue
                if obj.get("usage"): usage = obj["usage"]
                if obj.get("timings"): timings = obj["timings"]
                for ch in obj.get("choices", []):
                    d = ch.get("delta") or {}
                    # gemma4 emits reasoning_content before (or instead of) content. Both are
                    # generated tokens: counting only `content` reported ttft=None on every
                    # agentic prompt in the first self-check. Ollama calls it `reasoning`.
                    generated = False
                    if d.get("reasoning_content") or d.get("reasoning"):
                        reasoning.append(d.get("reasoning_content") or d.get("reasoning"))
                        generated = True
                    if d.get("content"):
                        text.append(d["content"]); generated = True
                    for tc in d.get("tool_calls") or []:
                        generated = True
                        i = tc.get("index", 0)
                        slot = tool_calls.setdefault(i, {"name": "", "arguments": ""})
                        fn = tc.get("function") or {}
                        if fn.get("name"): slot["name"] += fn["name"]
                        if fn.get("arguments"): slot["arguments"] += fn["arguments"]
                    if generated:
                        now = time.time()
                        if ttft is None: ttft = now - t0
                        last = now - t0; n_chunks += 1; chunk_times.append(last)
        total = time.time() - t0
        return {"ttft_s": ttft, "last_s": last, "total_s": total, "text": "".join(text),
                "reasoning": "".join(reasoning),
                "tool_calls": [tool_calls[k] for k in sorted(tool_calls)],
                "usage": usage or {}, "timings": timings, "chunks": n_chunks,
                "chunk_times": chunk_times}

    def stop(self, grace=30):
        # take the footprint before the process exits
        self.phys_footprint_mib = (phys_footprint_mib(self.proc.pid)
                                   if self.proc and self.proc.poll() is None else 0.0)
        self.peak_wired_mib = self.rss.peak_wired_mib if self.rss else 0.0
        self.base_wired_mib = self.rss.base_wired_mib if self.rss else 0.0
        peak = self.rss.stop() if self.rss else 0
        if self.proc and self.proc.poll() is None:
            group = self.process_group and os.name != "nt"
            if os.name == "nt":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            elif group:
                # The whole session: terminate() alone leaves a forked child on the port.
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            else:
                self.proc.send_signal(signal.SIGINT)
            for _ in range(grace * 10):
                if self.proc.poll() is not None: break
                time.sleep(0.1)
            if self.proc.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                                   capture_output=True)
                elif group:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                else:
                    self.proc.kill()
        if self.log: self.log.close()
        return peak
