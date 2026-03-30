# Virtual Fitter API Deployment

## Railway API Service

Deploy the FastAPI service from:

`side-projects/virtual-fitter/render-service`

### Required Railway Variables

| Variable | Purpose | Example |
| --- | --- | --- |
| `REPLICATE_API_TOKEN` | Replicate API key for SAM2 and FLUX | `r8_xxxxx` |
| `ENVIRONMENT` | Runtime environment | `production` |
| `DEBUG` | Enable docs and extra debug pages | `false` |
| `PORT` | Railway injected port | `8080` |
| `VF_MEDIA_BASE_URL` | Public media base for generated previews | `https://your-api.up.railway.app/media` |
| `SHOPIFY_ADMIN_ACCESS_TOKEN` | Shopify Admin API token for product metafields | `shpat_xxxxx` |
| `SHOPIFY_SHOP_DOMAIN` | Default Shopify domain | `kits-uk.myshopify.com` |
| `CORS_ORIGINS` | Comma-separated allowed origins | `https://kits.style,https://www.kits.style,https://kits-uk.myshopify.com,http://127.0.0.1:9292,http://localhost:9292` |

### Public Domain

1. Open Railway dashboard.
2. Select the Virtual Fitter API service.
3. Go to `Settings -> Networking -> Public Networking`.
4. Generate a public domain.
5. Use that domain for `VF_MEDIA_BASE_URL`.

### Health Checks

```bash
curl https://YOUR-RAILWAY-URL/v1/health
curl https://YOUR-RAILWAY-URL/v1/render-modes
```

## Remix Proxy Configuration

The Shopify storefront should keep calling the Remix proxy path:

`/apps/virtual-fitter/api`

Do not hardcode the Railway URL directly into the storefront unless you intentionally want cross-origin browser requests.

Instead, set one of these variables on the Remix app:

| Variable | Purpose |
| --- | --- |
| `VIRTUAL_FITTER_BACKEND_URL` | Preferred backend URL for the proxy |
| `RENDER_SERVICE_URL` | Backward-compatible fallback name |

Example:

```bash
VIRTUAL_FITTER_BACKEND_URL=https://your-api.up.railway.app
```

## Shopify Theme Configuration

The theme snippet already defaults to:

`/apps/virtual-fitter/api`

If needed, you can override the API base with:

- shop metafield `custom.virtual_fitter_api_base`
- or `window.KSVirtualFitterOverrides.apiBase`

## Procfile

Use:

```bash
web: uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

## Troubleshooting

### CORS errors

- Confirm the storefront domains are included in `CORS_ORIGINS`.
- If using the Remix proxy path, browser CORS issues should be minimal because the browser talks to the same app origin.

### 404 from Shopify storefront

- Confirm the Remix app has routes for:
  - `/apps/virtual-fitter/api/render-jobs`
  - `/apps/virtual-fitter/api/render-jobs/:jobId`
  - `/apps/virtual-fitter/api/render-modes`

### 502 or 500 from Railway

- Check Railway logs.
- Confirm `REPLICATE_API_TOKEN` is set.
- Confirm `VF_MEDIA_BASE_URL` points to the public Railway URL.

### Missing product assets

- Confirm product metafields exist under the `virtual_fitter` namespace.
- For temporary testing, storefront-provided overlay or featured image fallback can keep the flow alive.
