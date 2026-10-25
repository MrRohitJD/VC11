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
| `AZURE_OPENAI_ENDPOINT` | `https://rg-openai-sea.cognitiveservices.azure.com/` |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | `text-embedding-3-large`|
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | `gpt-4.1-mini` |

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

## Google OAuth production verification

Try verifying the exact Heroku URL as a **URL-prefix property** in Google Search Console: `https://ai-powered-gmail-0007703734bc.herokuapp.com/`. Use a Google account that is an Owner or Editor of the OAuth Cloud project. Choose HTML file verification, download the unique file supplied by Search Console, and place it unchanged in `gmail-ai-dashboard/frontend/public/`. Deploy it and confirm that `https://ai-powered-gmail-0007703734bc.herokuapp.com/<verification-file-name>` serves the exact file publicly, then click Verify. Keep the file deployed so verification remains valid. A Domain property requires DNS ownership and is not appropriate for `herokuapp.com`.

Deploy and confirm the public homepage (`/`), privacy policy (`/privacy`), and Terms of Service (`/terms`) are readable in a private browser window. In Google Auth Platform, set the consent screen app name to **Gmail AI Dashboard** to match the homepage. Use `https://ai-powered-gmail-0007703734bc.herokuapp.com/` as the homepage URL, `https://ai-powered-gmail-0007703734bc.herokuapp.com/privacy` as the privacy policy URL, and `https://ai-powered-gmail-0007703734bc.herokuapp.com/terms` as the Terms of Service URL. The redirect URI can remain `https://ai-powered-gmail-0007703734bc.herokuapp.com/api/auth/callback`.

After Search Console says the URL-prefix property is verified, wait the 24 hours requested in Google's rejection notice before resubmitting. Google may still require a custom domain if it does not accept ownership verification for this hosted subdomain; in that case attach a domain you control to Heroku, verify it, and update the homepage, privacy policy, environment origins, and OAuth redirect URI together.

Declare the two Gmail scopes requested by the app: `https://www.googleapis.com/auth/gmail.readonly` and `https://www.googleapis.com/auth/gmail.send`. Review the published privacy policy against the actual production deployment and provider contracts before submitting it to Google. Google must separately approve the scopes; resolving these website issues alone does not remove the unverified-app warning.
