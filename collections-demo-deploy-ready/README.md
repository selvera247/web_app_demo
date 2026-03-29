# Collections Demo — Deploy Ready

## Files included
- `api.py`
- `requirements.txt`
- `render.yaml`
- `web/index.html`

## Put your CSVs here
Place these files in the `data/` folder:
- `aging_snapshot.csv`
- `customers.csv`
- `payments.csv`
- `disputes.csv`
- `communication_log.csv`

## Run locally
```bash
pip install -r requirements.txt
uvicorn api:app --reload
```

## Render deploy
1. Create a GitHub repo and upload this project.
2. In Render, create a new Web Service from the repo.
3. Render should detect `render.yaml`.
4. Confirm the start command is:
   `uvicorn api:app --host 0.0.0.0 --port $PORT`
5. Commit your `data/` folder for demo purposes.
6. Deploy and copy the backend URL.

## Cloudflare Pages deploy
1. Create a new Pages project.
2. Upload the `web/` folder or connect the repo and set the build output to `web`.
3. Before deploying, open `web/index.html`.
4. Replace:
   `const API_BASE = "https://YOUR-RENDER-SERVICE.onrender.com";`
   with your real Render backend URL.
5. Deploy Pages.

## Squarespace
Use a button to the Cloudflare Pages URL or embed with an iframe.

## Demo defaults
- `DEMO_MODE=true`
- `OLLAMA_ENABLED=false`

That keeps the public demo stable.
