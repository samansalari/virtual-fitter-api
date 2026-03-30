from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx

from ..config import get_settings
from .validation import MissingProductMetafields

logger = logging.getLogger(__name__)


@dataclass
class ProductRenderAssets:
    product_id: str
    variant_id: str
    product_title: str
    variant_title: Optional[str]
    product_handle: str
    product_type: str
    placement_zone: str
    overlay_url: Optional[str]
    mask_url: Optional[str]
    anchors: dict
    compatible_models: list[str] = field(default_factory=list)
    render_prompt: str = ""


PRODUCT_KEYS = (
    "product_type",
    "placement_zone",
    "overlay_asset",
    "mask_template",
    "placement_anchors",
    "compatible_models",
    "render_prompt",
)


def _normalize_gid(resource: str, raw_id: str) -> str:
    if raw_id.startswith("gid://"):
        return raw_id
    return f"gid://shopify/{resource}/{raw_id}"


def _extract_metafield_url(field: Optional[dict]) -> Optional[str]:
    if not field:
        return None

    reference = field.get("reference")
    if isinstance(reference, dict):
        image = reference.get("image")
        if isinstance(image, dict) and image.get("url"):
            return image["url"]
        if reference.get("url"):
            return reference["url"]
        preview = reference.get("preview")
        if isinstance(preview, dict):
            image = preview.get("image")
            if isinstance(image, dict) and image.get("url"):
                return image["url"]
    return None


def _extract_metafield_value(field: Optional[dict]) -> Optional[str]:
    if not field:
        return None
    value = field.get("value")
    return value if isinstance(value, str) and value != "" else None


def _extract_preview_url(node: Optional[dict]) -> Optional[str]:
    if not isinstance(node, dict):
        return None

    if node.get("url"):
        return node["url"]

    image = node.get("image")
    if isinstance(image, dict) and image.get("url"):
        return image["url"]

    preview = node.get("preview")
    if isinstance(preview, dict):
        preview_image = preview.get("image")
        if isinstance(preview_image, dict) and preview_image.get("url"):
            return preview_image["url"]

    return None


def _parse_anchors(raw_value: Optional[str]) -> dict:
    if not raw_value:
        return {"x_offset": 0.5, "y_offset": 0.5, "scale_factor": 1.0, "allow_mirror": True}
    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, dict):
            defaults = {"x_offset": 0.5, "y_offset": 0.5, "scale_factor": 1.0, "allow_mirror": True}
            defaults.update(parsed)
            return defaults
    except json.JSONDecodeError:
        pass
    raise MissingProductMetafields("The placement anchor metafield is not valid JSON.")


def _parse_list(raw_value: Optional[str]) -> list[str]:
    if not raw_value:
        return []
    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _infer_from_overrides(overrides: dict[str, Optional[str]]) -> tuple[str, str]:
    placement_hint = (overrides.get("placement_hint") or "").lower()
    if "wheel" in placement_hint:
        return "wheel", "side_left"
    if "side" in placement_hint and "skirt" in placement_hint:
        return "side_skirt", "side_left"
    if "diffuser" in placement_hint:
        return "diffuser", "rear"
    if "mirror" in placement_hint:
        return "mirror_cap", "side_left"
    if "badge" in placement_hint:
        return "badge", "front"
    if "body" in placement_hint:
        return "body_kit", "full_body"
    return "spoiler", "rear"


async def _fetch_admin_payload(shop_domain: str, product_id: str, variant_id: str) -> dict:
    settings = get_settings()
    if not settings.shopify_admin_access_token:
        raise MissingProductMetafields("SHOPIFY_ADMIN_ACCESS_TOKEN is not configured for Virtual Fitter product lookups.")

    domain = shop_domain or settings.shopify_default_shop_domain
    if not domain:
        raise MissingProductMetafields("No Shopify shop domain was provided for Virtual Fitter product lookup.")

    endpoint = f"https://{domain}/admin/api/{settings.shopify_api_version}/graphql.json"
    query = """
    query VirtualFitterAssets($productId: ID!, $variantId: ID!) {
      product(id: $productId) {
        id
        title
        handle
        productType
        featuredMedia {
          ... on MediaImage { image { url } }
          ... on GenericFile { url }
          preview { image { url } }
        }
        product_type: metafield(namespace: "virtual_fitter", key: "product_type") { value }
        placement_zone: metafield(namespace: "virtual_fitter", key: "placement_zone") { value }
        overlay_asset: metafield(namespace: "virtual_fitter", key: "overlay_asset") {
          value
          reference {
            ... on MediaImage { image { url } }
            ... on GenericFile { url }
          }
        }
        mask_template: metafield(namespace: "virtual_fitter", key: "mask_template") {
          value
          reference {
            ... on MediaImage { image { url } }
            ... on GenericFile { url }
          }
        }
        placement_anchors: metafield(namespace: "virtual_fitter", key: "placement_anchors") { value }
        compatible_models: metafield(namespace: "virtual_fitter", key: "compatible_models") { value }
        render_prompt: metafield(namespace: "virtual_fitter", key: "render_prompt") { value }
      }
      variant: productVariant(id: $variantId) {
        id
        title
        image { url }
        product_type: metafield(namespace: "virtual_fitter", key: "product_type") { value }
        placement_zone: metafield(namespace: "virtual_fitter", key: "placement_zone") { value }
        overlay_asset: metafield(namespace: "virtual_fitter", key: "overlay_asset") {
          value
          reference {
            ... on MediaImage { image { url } }
            ... on GenericFile { url }
          }
        }
        mask_template: metafield(namespace: "virtual_fitter", key: "mask_template") {
          value
          reference {
            ... on MediaImage { image { url } }
            ... on GenericFile { url }
          }
        }
        placement_anchors: metafield(namespace: "virtual_fitter", key: "placement_anchors") { value }
        compatible_models: metafield(namespace: "virtual_fitter", key: "compatible_models") { value }
        render_prompt: metafield(namespace: "virtual_fitter", key: "render_prompt") { value }
      }
    }
    """
    variables = {
        "productId": _normalize_gid("Product", product_id),
        "variantId": _normalize_gid("ProductVariant", variant_id),
    }

    headers = {
        "X-Shopify-Access-Token": settings.shopify_admin_access_token,
        "Content-Type": "application/json",
    }
    timeout = settings.request_timeout_seconds
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(endpoint, headers=headers, json={"query": query, "variables": variables})
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise MissingProductMetafields("Shopify returned an error while loading Virtual Fitter assets.")
    return payload.get("data") or {}


def _merged_field(product: dict, variant: dict, key: str) -> Optional[str]:
    if key in {"overlay_asset", "mask_template"}:
        return _extract_metafield_url(variant.get(key)) or _extract_metafield_url(product.get(key))
    return _extract_metafield_value(variant.get(key)) or _extract_metafield_value(product.get(key))


def _fallback_overlay_url(product: dict, variant: dict, overrides: dict[str, Optional[str]]) -> Optional[str]:
    return (
        overrides.get("overlay_asset_url")
        or overrides.get("product_image_url")
        or overrides.get("featured_image_url")
        or _extract_preview_url(variant.get("image"))
        or _extract_preview_url(product.get("featuredMedia"))
    )


async def get_product_image_url(
    shop_domain: str,
    product_id: str,
    variant_id: Optional[str] = None,
    overrides: Optional[dict[str, Optional[str]]] = None,
) -> Optional[str]:
    overrides = overrides or {}
    if overrides.get("product_image_url") or overrides.get("featured_image_url"):
        return overrides.get("product_image_url") or overrides.get("featured_image_url")
    if not shop_domain:
        return None

    url = f"https://{shop_domain}/products.json"
    timeout = get_settings().request_timeout_seconds
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, params={"limit": 250}, headers={"Accept": "application/json"})
            response.raise_for_status()
    except Exception:
        return None

    payload = response.json()
    for product in payload.get("products", []):
        if str(product.get("id")) == str(product_id):
            images = product.get("images") or []
            if images and images[0].get("src"):
                return str(images[0]["src"])
    return None


async def fetch_product_render_assets(
    shop_domain: str,
    product_id: str,
    variant_id: str,
    overrides: Optional[dict[str, Optional[str]]] = None,
) -> ProductRenderAssets:
    overrides = overrides or {}
    logger.info(
        "Fetching product render assets: product_id=%s variant_id=%s shop_domain=%s override_overlay=%s",
        product_id,
        variant_id,
        shop_domain,
        bool(overrides.get("overlay_asset_url")),
    )

    try:
        payload = await _fetch_admin_payload(shop_domain, product_id, variant_id)
        product = payload.get("product") or {}
        variant = payload.get("variant") or {}
    except MissingProductMetafields as exc:
        logger.warning("Shopify asset lookup fell back to overrides: %s", exc)
        if not overrides:
            raise
        inferred_product_type, inferred_zone = _infer_from_overrides(overrides)
        overlay_url = overrides.get("overlay_asset_url") or overrides.get("product_image_url") or overrides.get("featured_image_url")
        return ProductRenderAssets(
            product_id=product_id,
            variant_id=variant_id,
            product_title=overrides.get("product_title") or "Unknown product",
            variant_title=overrides.get("variant_title"),
            product_handle=overrides.get("product_handle") or "",
            product_type=inferred_product_type,
            placement_zone=inferred_zone,
            overlay_url=overlay_url,
            mask_url=overrides.get("mask_asset_url"),
            anchors={"x_offset": 0.5, "y_offset": 0.5, "scale_factor": 1.0, "allow_mirror": True},
            compatible_models=[],
            render_prompt=overrides.get("placement_hint") or "",
        )

    product_type = _merged_field(product, variant, "product_type") or product.get("productType")
    placement_zone = _merged_field(product, variant, "placement_zone")
    overlay_url = _merged_field(product, variant, "overlay_asset") or _fallback_overlay_url(product, variant, overrides)
    if not overlay_url:
        overlay_url = await get_product_image_url(shop_domain, product_id, variant_id, overrides=overrides)
    mask_url = _merged_field(product, variant, "mask_template")
    anchors = _parse_anchors(_merged_field(product, variant, "placement_anchors"))
    compatible_models = _parse_list(_merged_field(product, variant, "compatible_models"))
    render_prompt = _merged_field(product, variant, "render_prompt") or ""

    if not product_type:
        inferred_product_type, _ = _infer_from_overrides(overrides)
        product_type = inferred_product_type
    if not placement_zone:
        _, inferred_zone = _infer_from_overrides(overrides)
        placement_zone = inferred_zone

    if not product_type or not placement_zone:
        logger.error(
            "Missing required virtual fitter product config: product_type=%s placement_zone=%s overlay_url=%s",
            product_type,
            placement_zone,
            bool(overlay_url),
        )
        raise MissingProductMetafields("Required Virtual Fitter product configuration is missing for this product.")

    return ProductRenderAssets(
        product_id=product_id,
        variant_id=variant_id,
        product_title=product.get("title") or overrides.get("product_title") or "Unknown product",
        variant_title=variant.get("title") or overrides.get("variant_title"),
        product_handle=product.get("handle") or overrides.get("product_handle") or "",
        product_type=product_type,
        placement_zone=placement_zone,
        overlay_url=overlay_url,
        mask_url=mask_url,
        anchors=anchors,
        compatible_models=compatible_models,
        render_prompt=render_prompt,
    )
