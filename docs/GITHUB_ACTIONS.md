# GitHub Actions CI/CD

The workflow at `.github/workflows/ci-cd.yml` runs backend tests from all three
test directories, frontend tests and the Vite build, then builds the root Dockerfile.
Pull requests and pushes to `devLG` and `heroku_deploy_dev_lg` run these checks.
Manual runs perform checks only.

## One-time setup

1. In GitHub Settings → Environments, create `production`. Restrict deployment
   branches to `heroku_deploy_dev_lg`. Add required reviewers if desired.
2. Add `HEROKU_API_KEY` as a production environment secret (or repository Actions
   secret). Obtain the credential using `heroku auth:token`; never commit it.
3. Confirm that Heroku app `ai-powered-gmail` uses the `container` stack:
   `heroku apps:info --app ai-powered-gmail`. If needed, set it with
   `heroku stack:set container --app ai-powered-gmail`.
4. Keep application credentials and DATABASE_URL in Heroku config vars as
   described in `DEPLOY_HEROKU.md`. Tests do not receive production credentials.
5. Push the workflow and inspect its first run in GitHub → Actions. Once checks
   appear, require `Backend tests`, `Frontend tests and build`, and
   `Production Docker build` in branch protection for the deployment branch.

## Deployment behavior

A push to `heroku_deploy_dev_lg` deploys the tested commit through Heroku Git
after all checks pass. **Adding this workflow to that branch enables automatic
production deployment.** PR and manual runs never deploy.

Heroku builds the root Dockerfile again using `heroku.yml`; the CI image is a
build check, not a promoted artifact. Dependencies are not fully pinned, so the
two builds are not guaranteed to be identical. The existing release phase runs
database reconciliation, Alembic migrations, and demo mailbox seeding. A failed
release command blocks activation. GitHub then checks the public `/health` URL.
That endpoint checks HTTP availability, not all database or provider connections.

Workflow runs on the same branch are serialized. GitHub can replace pending
runs with newer pending runs; this is not a queue guaranteeing every commit deploys.
Deployment uses a temporary Git authentication helper and never force-pushes Heroku history.

## Troubleshooting

- Failed tests or Docker builds: inspect the failed Actions step; no deployment
  occurs. Backend JUnit results are attached as an artifact when available.
- Missing secret: set `HEROKU_API_KEY` in the environment or repository.
- Git push rejected as non-fast-forward: reconcile the deployment branch with
  existing Heroku history. Do not add an automatic force push to the workflow.
- Deployment or health failure: inspect `heroku logs --tail --app ai-powered-gmail`.
  A failed health check marks the workflow failed but does not roll back a release.
- To change the target app or public URL, update the deployment job and confirm
  the target app's stack and runtime configuration first.

References: [GitHub Python CI](https://docs.github.com/en/actions/tutorials/build-and-test-code/python),
[Heroku Git](https://devcenter.heroku.com/articles/git),
[Heroku container builds](https://devcenter.heroku.com/articles/build-docker-images-heroku-yml).
