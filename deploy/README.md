# Server Deployment

The server runtime is intentionally separated from training data and local development dependencies.

Upload or sync only these application paths:

```text
src/
scripts/
third_party/Kronos/
config/
frontend/dist/
requirements.txt
deploy/
README.md
A股数据/股票列表.csv
```

The existing server keeps its own `data/`, `models/`, `logs/`, and research artifacts. `third_party/Kronos/` contains the pinned model implementation and is deployed with application code; it never contains model weights. Do not replace the runtime data or model directories during a web deployment.

Create a reproducible release archive on Windows:

```powershell
.\deploy\package_release.ps1
```

For an intentional runtime-data refresh, add `-IncludeDatabase`. Back up the
server database before extraction; this option includes only `data/kronos.db`
and its A-share download metadata, never training datasets or model files.

## Full-market daily quotes

The stock search and detail pages can show the latest **daily** OHLCV available
on the server. They do not supply intraday or real-time quotes. To enable
full-market browsing without extracting an archive, upload only:

```text
A股数据/股票列表.csv
A股数据/daily_qfq.zip   # preferred for chart display
```

Place them on the server at the same paths under `/root/kronos-strategy/`.
`daily.zip` is also supported and is sufficient when it is already present at
`data/source/daily.zip`; do not upload both archives solely for web browsing.
The weekly, monthly, post-adjusted archives, and the local extracted `daily_*`
directories are not required by this feature.

The API is available after restarting the service:

```bash
curl 'http://127.0.0.1:8000/api/v1/index/list?type=stock&keyword=000001'
curl 'http://127.0.0.1:8000/api/v1/index/000001?days=250'
```

On the server:

```bash
cd /root/kronos-strategy
cp deploy/production.env.example .env.production
chmod 600 .env.production
# Edit .env.production and set a long random KRONOS_ADMIN_PASSWORD.

sudo cp deploy/kronos-strategy.service /etc/systemd/system/kronos-strategy.service
sudo systemctl daemon-reload
sudo systemctl enable --now kronos-strategy
sudo systemctl status kronos-strategy
```

For GPU containers without a running systemd PID 1, use the tracked background
process instead:

```bash
cd /root/kronos-strategy
chmod 750 deploy/start_server.sh deploy/stop_server.sh
./deploy/start_server.sh
python scripts/verify_deployment.py
```

The PID is written to `run/kronos-web.pid` and logs are written to
`logs/kronos-web.log`. Stop it with `./deploy/stop_server.sh`.

The FastAPI service serves both the built React SPA and `/api/v1/*` from port `8000`. Check `/health` after deployment. A public URL additionally requires the cloud provider's HTTP port mapping or reverse proxy configuration; the SSH port is not an HTTP port.
