# Deploy Gmail AI Dashboard on Heroku

The `ai-powered-gmail` app runs in a single web container containing both the React frontend and FastAPI backend. Heroku Postgres stores email data, chat history, and 3072-dimensional `halfvec` embeddings via pgvector.

This deployment is based on the **`devLG`** branch, which contains the latest LangGraph mailbox orchestration, Gmail MCP integration, updated vector database migrations, and chat rate limiting.

> **Note:** Do not use the `main` branch for deployment. Always use `devLG` as the base branch, deployed via the `heroku_deploy_dev_lg` branch.

## Heroku Settings

The existing Postgres add-on owns `DATABASE_URL`. Leave this setting attached to the add-on so credential rotations work automatically. The backend accepts Heroku's `postgres://` URL and connects via `psycopg`.

Set these config vars on `ai-powered-gmail`:

| Name | Value |
| --- | --- |
| `ENVIRONMENT` | `production` |
| `RAG_VECTOR_BACKEND` | `postgres` |
| `GMAIL_SYNC_LIMIT` | `500` |
| `DATABASE_POOL_SIZE` | `2` |
| `DATABASE_MAX_OVERFLOW` | `3` |
| `LANGSMITH_TRACING` | `false` |
| `FRONTEND_URL` | `https://ai-powered-gmail-0007703734bc.herokuapp.com` |
| `BACKEND_URL` | `https://ai-powered-gmail-0007703734bc.herokuapp.com` |
| `OAUTH_GOOGLE_REDIRECT_URI` | `https://ai-powered-gmail-0007703734bc.herokuapp.com/api/auth/callback` |
| `SECURE_COOKIES` | `true` |
| `AZURE_OPENAI_ENDPOINT` | Your Azure OpenAI resource endpoint, e.g. `https://<RESOURCE>.openai.azure.com/` |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Name of your 3072-dimensional `text-embedding-3-large` deployment |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Name of your chat deployment (defaults to `gpt-4.1-mini`) |

Keep `SESSION_SECRET`, `AZURE_OPENAI_API_KEY`, `OAUTH_GOOGLE_CLIENT_ID`, and `OAUTH_GOOGLE_CLIENT_SECRET` as Heroku config vars. Do not commit them.

Ensure `AZURE_OPENAI_ENDPOINT` is set in Heroku config vars and that `AZURE_OPENAI_API_KEY` has no trailing quotes or whitespace.

In Google Cloud Console, add `https://ai-powered-gmail-0007703734bc.herokuapp.com/api/auth/callback` to the OAuth web client's **Authorized redirect URIs**.

## Build and Release

The Heroku app uses the `container` stack. The repository root contains `heroku.yml` and `Dockerfile`:
1. **Frontend build:** Builds Vite React assets and outputs to `/app/frontend/dist`.
2. **Backend build:** Installs dependencies from `requirements-heroku.txt`.
3. **Release phase:** Runs `sh scripts/heroku_release.sh`, which runs `alembic upgrade head` and seeds the demo mailbox with `python scripts/seed_demo_mailbox.py`.
4. **Web dyno:** Starts Uvicorn serving both FastAPI API endpoints and frontend static assets.

### Deploying Changes

If using GitHub connected deployments on Heroku Dashboard:
- Connect the **`heroku_deploy_dev_lg`** branch (based on `devLG`) and trigger manual or automatic deploys. Do not connect `main`.

If deploying via Heroku Git CLI:
```powershell
heroku git:remote -a ai-powered-gmail
git push heroku heroku_deploy_dev_lg:main --force
```

After release, open `https://ai-powered-gmail-0007703734bc.herokuapp.com/health` to confirm the service is healthy.
