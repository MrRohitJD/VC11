# Deploy Gmail AI Dashboard on Heroku

The `ai-powered-gmail` app runs in a single web container containing both the React frontend and FastAPI backend. Heroku Postgres stores email data, chat history, and 3072-dimensional `halfvec` embeddings via pgvector.

This deployment is based on the **`devLG`** branch, which contains the latest LangGraph mailbox orchestration, Gmail MCP integration, updated vector database migrations, explicit database routing (`APP_DEFAULT`, `PERSONAL_ACTIVE`, `APP_FALLBACK`), SSRF protection, encryption key isolation, and chat rate limiting.

> **Note:** Do not use the `main` branch for deployment. Always use `devLG` as the base branch, deployed via the `heroku_deploy_dev_lg` branch.

## Heroku Settings

The existing Postgres add-on owns `DATABASE_URL`. Leave this setting attached to the add-on so credential rotations work automatically. The backend accepts Heroku's `postgres://` URL and connects via `psycopg`.

Set these config vars on `ai-powered-gmail`:

| Name | Description / Value |
| --- | --- |
| `ENVIRONMENT` | `production` (enforces strict key entropy, cookie security, and SSRF restrictions) |
| `SESSION_SECRET` | Required. High-entropy random secret (min 32 chars). Protects signed session cookies. |
| `DEVELOPER_SETTINGS_ENCRYPTION_KEY` | Required. High-entropy secret (min 32 chars). Distinct from `SESSION_SECRET`. Encrypts BYODB database URLs and personal API keys. |
| `DEVELOPER_SETTINGS_LEGACY_ENCRYPTION_KEY` | Optional. If rotating keys on an existing deployment, set this to the previous secret to decrypt legacy settings without data loss. |
| `DEVELOPER_DATABASE_ALLOW_PRIVATE_NETWORKING` | `false` (blocks SSRF to AWS/GCP metadata `169.254.169.254`, loopback, and RFC 1918 private subnets) |
| `DEMO_IP_DAILY_LIMIT` | `25` (prevents demo quota evasion via cookie clearing) |
| `DEMO_GLOBAL_DAILY_LIMIT` | `500` (global daily demo query budget) |
| `RAG_VECTOR_BACKEND` | `postgres` |
| `DATABASE_POOL_SIZE` | `2` |
| `DATABASE_MAX_OVERFLOW` | `3` |
| `LANGSMITH_TRACING` | `false` |
| `FRONTEND_URL` | `https://mail-assistant.rohit-jadhav.dev` |
| `BACKEND_URL` | `https://mail-assistant.rohit-jadhav.dev` |
| `OAUTH_GOOGLE_REDIRECT_URI` | `https://mail-assistant.rohit-jadhav.dev/api/auth/callback` |
| `SECURE_COOKIES` | `true` |
| `AZURE_OPENAI_ENDPOINT` | `https://rg-openai-sea.cognitiveservices.azure.com/` |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | `text-embedding-3-large`|
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | `gpt-4.1-mini` |

Keep `SESSION_SECRET`, `DEVELOPER_SETTINGS_ENCRYPTION_KEY`, `AZURE_OPENAI_API_KEY`, `OAUTH_GOOGLE_CLIENT_ID`, and `OAUTH_GOOGLE_CLIENT_SECRET` as Heroku config vars. Do not commit them.

### Generating High-Entropy Secrets

Generate separate cryptographic keys for `SESSION_SECRET` and `DEVELOPER_SETTINGS_ENCRYPTION_KEY`:

```bash
# Generate SESSION_SECRET (44 chars URL-safe)
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate DEVELOPER_SETTINGS_ENCRYPTION_KEY (distinct key)
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Apply them via the Heroku CLI:
```bash
heroku config:set SESSION_SECRET="<generated-session-secret>" -a ai-powered-gmail
heroku config:set DEVELOPER_SETTINGS_ENCRYPTION_KEY="<generated-developer-key>" -a ai-powered-gmail
```

Ensure `AZURE_OPENAI_ENDPOINT` is set in Heroku config vars and that `AZURE_OPENAI_API_KEY` has no trailing quotes or whitespace.

In Google Cloud Console, remove the old Heroku callback URI and add `https://mail-assistant.rohit-jadhav.dev/api/auth/callback` to the OAuth web client's **Authorized redirect URIs**. The value in `OAUTH_GOOGLE_REDIRECT_URI` must match this URI exactly.

---

## Deployment-Owner Verification Checklist

Run these safe verification commands before and after deploying to production:

### 1. Verify Configuration Entropy and Distinctness (Without Logging Secret Values)
```powershell
# Verify SESSION_SECRET character count is at least 32
$secLen = (heroku config:get SESSION_SECRET -a ai-powered-gmail).Trim().Length
if ($secLen -ge 32) { Write-Host "SESSION_SECRET length OK: $secLen chars" } else { Write-Error "SESSION_SECRET too short!" }

# Verify DEVELOPER_SETTINGS_ENCRYPTION_KEY character count is at least 32
$devLen = (heroku config:get DEVELOPER_SETTINGS_ENCRYPTION_KEY -a ai-powered-gmail).Trim().Length
if ($devLen -ge 32) { Write-Host "DEVELOPER_SETTINGS_ENCRYPTION_KEY length OK: $devLen chars" } else { Write-Error "DEVELOPER_SETTINGS_ENCRYPTION_KEY too short!" }

# Verify keys are independent
$sec = (heroku config:get SESSION_SECRET -a ai-powered-gmail).Trim()
$dev = (heroku config:get DEVELOPER_SETTINGS_ENCRYPTION_KEY -a ai-powered-gmail).Trim()
if ($sec -ne $dev) { Write-Host "Keys are properly isolated and distinct." } else { Write-Error "CRITICAL: Keys must not be identical!" }
```

### 2. Verify Startup & Health Check
```powershell
# Test health endpoint returns status ok
curl -i https://mail-assistant.rohit-jadhav.dev/health
```

### 3. Verify Database Routing Headers
```powershell
# Authenticated API calls return X-Database-Mode header (APP_DEFAULT, PERSONAL_ACTIVE, or APP_FALLBACK)
curl -i https://mail-assistant.rohit-jadhav.dev/api/developer
```

---

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

After release, open `https://mail-assistant.rohit-jadhav.dev/health` to confirm the service is healthy.

## Google OAuth production verification

Verify `mail-assistant.rohit-jadhav.dev` in Google Search Console before submitting the OAuth app for verification. Use a Google account that is an Owner or Editor of the OAuth Cloud project. Keep the verification file or DNS record deployed.

Deploy and confirm the public homepage (`/`), privacy policy (`/privacy`), and Terms of Service (`/terms`) are readable in a private browser window. In Google Auth Platform, set the consent screen app name to **Gmail AI Dashboard** to match the homepage. Use `https://mail-assistant.rohit-jadhav.dev/` as the homepage URL, `https://mail-assistant.rohit-jadhav.dev/privacy` as the privacy policy URL, and `https://mail-assistant.rohit-jadhav.dev/terms` as the Terms of Service URL. The redirect URI must be `https://mail-assistant.rohit-jadhav.dev/api/auth/callback`.

After Search Console says the URL-prefix property is verified, wait the 24 hours requested in Google's rejection notice before resubmitting. Google may still require a custom domain if it does not accept ownership verification for this hosted subdomain; in that case attach a domain you control to Heroku, verify it, and update the homepage, privacy policy, environment origins, and OAuth redirect URI together.

Declare the two Gmail scopes requested by the app: `https://www.googleapis.com/auth/gmail.readonly` and `https://www.googleapis.com/auth/gmail.send`. Review the published privacy policy against the actual production deployment and provider contracts before submitting it to Google. Google must separately approve the scopes; resolving these website issues alone does not remove the unverified-app warning.
