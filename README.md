# Model Runner

A small FastAPI server that runs generative models on a GPU behind an HTTP API.
Jobs go into a queue, and a single worker runs them one at a time on the GPU.

| Model id | Task | Status |
|---|---|---|
| `wan2.2-ti2v-5b` | text-to-video, or image-to-video when you send `image` | enabled |
| `sdxl-placeholder` | image | disabled (stub, not implemented yet) |

Models are listed in [config/models.yaml](config/models.yaml).

---

## 1. Requirements

- Linux with Python 3.12+ (uv installs Python for you if it's missing)
- [uv](https://docs.astral.sh/uv/)
- **To generate video:** an NVIDIA GPU with a working driver (`nvidia-smi` must run).
  A 24 GB card works with `CPU_OFFLOAD=true`. On cards with about 40 GB or more (A40, A100, H100), set `CPU_OFFLOAD=false` for speed.
- Enough disk space for the model weights (they go in the Hugging Face cache, `~/.cache/huggingface` by default) and for the generated videos (`outputs/`).

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 2. Install

```bash
git clone <repo-url> model-runner
cd model-runner

# API only (no GPU libraries); enough to run the server and the tests
uv sync

# On the GPU machine: add torch, diffusers and the other GPU dependencies
uv sync --extra gpu
```

Config is optional. Every setting has a default. To change any of them:

```bash
cp .env.example .env
# then edit .env
```

| Variable | Default | What it does |
|---|---|---|
| `APP_NAME` | `Model Runner` | Title shown in the API docs |
| `MODELS_CONFIG_PATH` | `config/models.yaml` | Model list |
| `OUTPUTS_DIR` | `outputs` | Where generated videos are written; also served at `/outputs/...` |
| `DEFAULT_IMAGE_MODEL` | `sdxl-placeholder` | Used when an image request has no `model` |
| `DEFAULT_VIDEO_MODEL` | `wan2.2-ti2v-5b` | Used when a video request has no `model` |
| `CPU_OFFLOAD` | `true` | Moves model parts between CPU and GPU so it fits on 24 GB. Set it to `false` on 40 GB+ GPUs. |
| `HF_HOME` | `~/.cache/huggingface` | (Hugging Face setting) Where model weights are downloaded |

---

## 3. Run it

### Local / development

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Then open:

- http://localhost:8000/docs: interactive API docs (Swagger), where you can try every endpoint
- http://localhost:8000/health should return `{"status":"ok"}`

### On a server

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- `--host 127.0.0.1` accepts connections from the machine itself only. Use this with an SSH tunnel (section 5, option A).
- `--host 0.0.0.0` accepts connections from the network. You also need to open the port (section 5, option B).

> **Always run a single worker.** Do not pass `--workers N`.
> The job queue lives in memory, so each extra worker would load its own copy of the model onto the GPU,
> and `GET /v1/jobs/{id}` would return 404 whenever the request reached a different worker.

---

## 4. Deploy (keep it running after you log out)

### Option 1: tmux (quickest)

```bash
tmux new -s runner
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
# detach: Ctrl+B then D       reattach: tmux attach -t runner
```

### Option 2: systemd (restarts on crash and on reboot)

Create `/etc/systemd/system/model-runner.service`, replacing the user and paths with yours:

```ini
[Unit]
Description=Model Runner
After=network.target

[Service]
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/model-runner
ExecStart=/home/YOUR_USER/model-runner/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now model-runner   # start now and on every boot
sudo systemctl status model-runner         # is it running?
journalctl -u model-runner -f              # follow the logs
sudo systemctl restart model-runner        # after a git pull or a .env change
```

### Updating

```bash
git pull
uv sync --extra gpu
sudo systemctl restart model-runner        # or Ctrl+C and start it again in tmux
```

Queued and running jobs are lost on restart. Finished videos in `outputs/` stay on disk.

---

## 5. Open the port

> ⚠️ **The API has no authentication.** Anyone who can reach the port can queue GPU jobs and download every generated video.
> Prefer option A. If you use option B, allow only your own IP address.

### Option A: SSH tunnel (recommended, no port is opened)

On the server, bind to localhost only:

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

On your own computer:

```bash
ssh -N -L 8000:localhost:8000 YOUR_USER@SERVER_IP
```

Now open http://localhost:8000/docs on your computer. To close it, press Ctrl+C in the ssh terminal.

### Option B: firewall (ufw)

Start the server with `--host 0.0.0.0`, then:

```bash
# Allow SSH FIRST so enabling the firewall doesn't lock you out
sudo ufw allow OpenSSH

# Allow port 8000 only from your IP (find it with: curl ifconfig.me)
sudo ufw allow from YOUR_IP to any port 8000 proto tcp

# Or allow everyone (not recommended, see the warning above)
# sudo ufw allow 8000/tcp

sudo ufw enable
sudo ufw status numbered
```

Then open `http://SERVER_IP:8000/docs`.

**Cloud GPU (AWS, GCP, RunPod, Vast.ai, etc.):** the provider also has its own firewall, which may be called a security group, firewall rule, or "exposed ports".
Add an inbound TCP rule for port 8000 there as well, limited to your IP if the provider allows it.

---

## 6. Test that it works

### Automated tests (no GPU needed)

```bash
uv run pytest -q
```

[tests/test_api.py](tests/test_api.py) checks health, the model list, request validation and the error codes, the job queue, and every stats section.
It doesn't load the real video model.

Lint:

```bash
uv run ruff check .
```

### Smoke test a running server

```bash
URL=http://localhost:8000

curl $URL/health                 # {"status":"ok"}
curl $URL/v1/models              # lists both models
curl $URL/v1/stats/gpu           # nvidia-smi info; shows an error if no GPU is visible
```

### Generate a video (GPU)

The first job downloads the model weights and loads them, so it is much slower than later jobs.
For a quick test, use few frames and steps:

```bash
curl -s -X POST $URL/v1/generate/video \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a red fox running through snow", "num_frames": 25, "steps": 10, "width": 640, "height": 352}'
```

You get back `202` and a job, for example `{"id": "abc123...", "status": "queued", ...}`. Check on it with:

```bash
curl -s $URL/v1/jobs/JOB_ID
```

`progress` shows the current phase (`loading model`, `denoising` with step and ETA, then `encoding video`).
When `status` is `completed`, `result.video_url` is the video's path:

```bash
curl -o out.mp4 $URL/outputs/FILE_NAME.mp4
```

**Image-to-video:** send the first frame as plain base64, without a `data:` prefix.
Write the request to a file, because a big image is too long for a command-line argument:

```bash
printf '{"prompt": "the cat starts walking", "image": "%s"}' "$(base64 -w0 first_frame.png)" > req.json
curl -s -X POST $URL/v1/generate/video -H 'Content-Type: application/json' -d @req.json
```

The image's aspect ratio decides the video's shape. `width` × `height` only sets the total pixel area.

---

## 7. Stop the server and close the port

Closing the port takes up to three steps, depending on how you opened it:

1. **Stop the process** that is listening on the port. Once nothing is listening, connections are refused.
2. **Remove the firewall rule** (only if you used section 5, option B). If you skip this, anything you start on port 8000 later is reachable from the network again.
3. **Close the SSH tunnel** (only if you used section 5, option A).

The commands below use port 8000. If you started the server with a different `--port`, use that number instead.

### Step 1: find what is using the port

```bash
ss -ltnp 'sport = :8000'
```

```
State  Recv-Q Send-Q Local Address:Port Peer Address:Port Process
LISTEN 0      2048         0.0.0.0:8000      0.0.0.0:*    users:(("python",pid=43383,fd=3),("uvicorn",pid=43381,fd=3))
```

- **Nothing below the header:** nothing is listening. The port is already closed.
- `0.0.0.0:8000` means the port is reachable from the network. `127.0.0.1:8000` means it is reachable from this machine only.
- `pid=` gives the process ids. A server started with `--reload` has two processes (the reloader and the server), and you have to stop both.
- **No `users:(...)` part:** the process belongs to another user. Run the command again with `sudo`.

`sudo lsof -i :8000` and `fuser -v 8000/tcp` show the same information.

### Step 2: stop the process

| How it was started | How to stop it |
|---|---|
| In a terminal | Press `Ctrl+C` in that terminal. If it doesn't exit, press it again. |
| In tmux | Run `tmux attach -t runner`, then press `Ctrl+C`. To stop it without attaching: `tmux kill-session -t runner` |
| systemd | `sudo systemctl stop model-runner`. Also run `sudo systemctl disable model-runner`, or it starts again at the next boot. |
| Unknown, or the terminal is gone | `fuser -k -TERM 8000/tcp` stops every process on the port, including both `--reload` processes. Or run `kill PID` for each pid from step 1. |
| Still running after that | Force it with `fuser -k 8000/tcp` or `kill -9 PID`. The process won't get a chance to shut down cleanly. |

> **Don't `kill` the systemd service.** With `Restart=on-failure`, systemd treats a force kill as a crash and restarts the server a few seconds later, so the port opens again. Use `systemctl stop`.

- If the process belongs to another user, put `sudo` in front of `fuser` or `kill`.
- If you get `fuser: command not found`, install it with `sudo apt install psmisc`, or use `kill PID` instead.
- Stopping the server loses the running job and everything in the queue. Finished videos in `outputs/` stay on disk. The GPU memory is freed when the process exits, which you can confirm with `nvidia-smi`.

Run step 1 again. If it prints nothing, the port is closed.

### Step 3: remove the firewall rule (ufw)

```bash
sudo ufw status numbered          # list the rules with their numbers
sudo ufw delete NUMBER            # delete the port 8000 rule by its number
```

The numbers change after every delete, so list the rules again before you delete the next one.
If the list has a port 8000 rule marked `(v6)`, delete it as well.

You can also delete a rule by repeating it exactly as you added it:

```bash
sudo ufw delete allow from YOUR_IP to any port 8000 proto tcp
sudo ufw delete allow 8000/tcp
```

Deleting the allow rule closes the port only if ufw blocks incoming traffic by default. To check:

```bash
sudo ufw status verbose           # should show: Default: deny (incoming)
```

If it shows `allow (incoming)` instead, block the port explicitly:

```bash
sudo ufw deny 8000/tcp
```

Keep the `OpenSSH` rule, or you can lock yourself out of the server.

If the server uses firewalld instead of ufw (Fedora, RHEL, Rocky):

```bash
sudo firewall-cmd --remove-port=8000/tcp --permanent
sudo firewall-cmd --reload
```

**Cloud GPU:** also delete the inbound port 8000 rule in the provider's dashboard. Depending on the provider, it is called a security group, a firewall rule, or "exposed ports".

### Close the SSH tunnel

The tunnel holds port 8000 on **your own computer**, not on the server. To close it, press `Ctrl+C` in the terminal where `ssh -N -L ...` is running.
If you started the tunnel in the background (`ssh -f`), stop it with:

```bash
pkill -f '8000:localhost:8000'
```

To check, run `ss -ltnp 'sport = :8000'` on your computer (on macOS, use `lsof -i :8000`).

### Check from another computer

```bash
nc -zv -w 3 SERVER_IP 8000
```

| Result | Meaning |
|---|---|
| `succeeded` / `open` | The port is still open. Go back to step 1. |
| `Connection refused` | Nothing is listening, but the firewall usually still lets traffic through. Do step 3. |
| Timed out | A firewall blocks the port, so it is closed. |

---

## 8. API reference

All endpoints are also documented at `/docs`.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/v1/models` | All configured models |
| POST | `/v1/generate/video` | Queue a video job → `202` + job |
| POST | `/v1/generate/image` | Queue an image job (the image model is a stub for now) |
| GET | `/v1/jobs?status=queued\|running\|completed\|failed` | All jobs, newest first |
| GET | `/v1/jobs/{id}` | One job: status, `position` in the queue, `progress`, `result`, `error` |
| GET | `/v1/stats` | Everything below at once |
| GET | `/v1/stats/{queue\|models\|gpu\|system\|disk\|info}` | One section only, cheap to poll |
| GET | `/outputs/{file}.mp4` | Download a generated video |

Video request fields ([app/schemas.py](app/schemas.py)):

| Field | Default | Limits |
|---|---|---|
| `prompt` | required | 1–10,000 characters |
| `negative_prompt` | none | ≤ 10,000 characters |
| `model` | `DEFAULT_VIDEO_MODEL` | |
| `image` | none | base64; send it for image-to-video |
| `num_frames` | 121 | 5–121 |
| `fps` | 24 | 1–60 |
| `width` × `height` | 1280 × 704 | 256–1280, multiples of 32 |
| `steps` | 50 | 1–100 |
| `guidance_scale` | 5.0 | 1–20 |
| `seed` | random | 0 to 2³²−1; the seed used is returned in `result` |

Error codes:

| Code | Meaning |
|---|---|
| `422` | Invalid input |
| `404` | Unknown model or job |
| `400` | The model doesn't do that task (e.g. a video model on `/generate/image`) |
| `503` | The model is disabled in `config/models.yaml` |

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| Job fails with CUDA out of memory | Set `CPU_OFFLOAD=true`, or lower `num_frames` / `width` × `height` |
| `/v1/stats/gpu` shows `nvidia-smi` not found | The NVIDIA driver isn't installed, or this machine has no GPU |
| The first job sits in `loading model` for a long time | The weights are downloading. Check progress with `du -sh ~/.cache/huggingface` |
| `Address already in use` | Something is already on port 8000. Section 7 shows how to find and stop it. Or pick another `--port` |
| Can't connect from another computer | Did you start with `--host 0.0.0.0`? Is the port open in ufw **and** in the cloud firewall? |
| Jobs disappeared | The server restarted. Jobs are only kept in memory; finished videos are still in `outputs/` |
| Disk filling up | Nothing deletes old videos. Clean up `outputs/` yourself, and see `/v1/stats/disk` |
