# KAGrobot Frontend (Vercel)

## Local development

1. Copy env file

```bash
cp .env.example .env.local
```

2. Fill your backend URL in `.env.local`

```env
NEXT_PUBLIC_API_BASE_URL=https://your-cloud-run-url
```

3. Install and start

```bash
npm install
npm run dev
```

## Vercel deployment

1. Import this repo into Vercel.
2. Set Root Directory to `frontend`.
3. Add environment variable:
   - `NEXT_PUBLIC_API_BASE_URL`: your Cloud Run base URL
4. Deploy.

## Backend CORS

Set backend env var `CORS_ALLOW_ORIGINS` (comma-separated), for example:

```env
CORS_ALLOW_ORIGINS=https://your-project.vercel.app,https://your-custom-domain.com
```
