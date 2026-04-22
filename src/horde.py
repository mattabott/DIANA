"""Minimal client for Stable Horde (https://stablehorde.net).

Main entry point:
  - generate_image(prompt, nsfw=None) -> bytes  (awaitable)

Flow:
  1. POST /api/v2/generate/async with prompt+params -> returns task id
  2. Poll GET /api/v2/generate/check/<id> until done: true
  3. GET /api/v2/generate/status/<id> -> image URL
  4. Download the URL and return the raw bytes

Raises HordeError on total timeout or submit errors.
"""
from __future__ import annotations

import asyncio
import logging
import httpx
from typing import Any

from src.config import CONFIG


log = logging.getLogger("diana-bot.horde")

API_BASE = "https://stablehorde.net/api/v2"
CLIENT_AGENT = "diana-bot/1.0"


class HordeError(Exception):
    pass


class CensorshipDetected(HordeError):
    """Horde applied its CSAM post-filter and returned a censorship banner
    instead of the generated photo. The returned image is typically much
    smaller than a real photo (<25KB)."""
    pass


# Size threshold below which we consider a file to be the "censorship banner".
# A real 384x576 JPEG is always >30KB; the banner is ~2-15KB.
CENSORSHIP_SIZE_THRESHOLD = 25_000


# Reasonable defaults for a portrait. Sizes/steps kept below the "upfront
# kudos" threshold so anonymous requests go through.
DEFAULT_PARAMS: dict[str, Any] = {
    "sampler_name": "k_euler_a",
    "cfg_scale": 7,
    "steps": 28,
    # Horde anonymous requires upfront kudos for any side > 576px or steps > 50.
    # 384x576 -> classic 2:3 portrait, multiple of 64 (SD-friendly), free tier.
    "width": 384,
    "height": 576,
    # Horde's minimum allowed seed_variation is 1 (minimal variation).
    "seed_variation": 1,
    "karras": True,
    "hires_fix": False,
    "clip_skip": 1,
    "n": 1,
}

# Negative prompt tuned to cut the "cinematic/artsy" look in favor of
# amateur smartphone photos. Added: bokeh, film grain, studio lights,
# plastic skin, airbrush etc. Extend if you still see too much "pro" look.
DEFAULT_NEGATIVE = (
    # stylized / non-photoreal
    "blurry, low quality, cartoon, anime, drawing, painting, illustration, "
    "3d render, CGI, digital art, concept art, artistic, cinematic, "
    # pro photography look
    "film grain, bokeh, shallow depth of field, dramatic lighting, "
    "studio lighting, professional photography, fashion photography, "
    "magazine cover, portrait photography, editorial, "
    # plastic/filtered skin
    "smooth skin, airbrushed, plastic skin, doll, flawless skin, "
    "instagram filter, beauty filter, oversaturated, HDR, "
    # wrong age (subjects looking too old)
    "old woman, mature woman, middle-aged, mother figure, matronly, older looking, "
    "wrinkles, crows feet, fine lines, sagging skin, grey hair, aging, elderly, "
    "tired face, dull skin, sunken cheeks, hollow eyes, "
    # body: no chubby / overweight, no excessively skinny
    "chubby, overweight, fat, plump, heavy-set, BBW, "
    "wide hips, thick thighs, double chin, "
    "anorexic, excessively thin, emaciated, "
    # anatomy
    "deformed, disfigured, bad anatomy, extra limbs, extra fingers, bad hands, "
    # basic safety (intentionally no child/teen/minor terms: Horde CSAM filter
    # flags them even inside the negative prompt)
    "text, watermark, signature, logo, ugly"
)


async def _submit(client: httpx.AsyncClient, payload: dict) -> str:
    r = await client.post(
        f"{API_BASE}/generate/async",
        json=payload,
        headers={
            "apikey": CONFIG.horde_api_key,
            "Client-Agent": CLIENT_AGENT,
        },
        timeout=30.0,
    )
    if r.status_code not in (200, 202):
        raise HordeError(f"submit failed: {r.status_code} {r.text[:300]}")
    data = r.json()
    task_id = data.get("id")
    if not task_id:
        raise HordeError(f"no id in response: {data}")
    log.info("horde task queued: %s", task_id)
    return task_id


async def _wait_for_done(client: httpx.AsyncClient, task_id: str, max_wait_s: float) -> None:
    poll_interval = 6.0  # seconds
    elapsed = 0.0
    last_log = 0.0
    while elapsed < max_wait_s:
        r = await client.get(
            f"{API_BASE}/generate/check/{task_id}",
            headers={"Client-Agent": CLIENT_AGENT},
            timeout=20.0,
        )
        if r.status_code != 200:
            raise HordeError(f"check failed: {r.status_code} {r.text[:200]}")
        data = r.json()
        if data.get("faulted"):
            raise HordeError(f"task faulted: {data}")
        if data.get("done"):
            return
        # Progress log every ~30s.
        if elapsed - last_log >= 30:
            log.info(
                "horde check %s: queue_position=%s wait_time=%ss processing=%s",
                task_id, data.get("queue_position"), data.get("wait_time"), data.get("processing"),
            )
            last_log = elapsed
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
    raise HordeError(f"timeout after {max_wait_s}s waiting for task {task_id}")


async def _fetch_result_url(client: httpx.AsyncClient, task_id: str) -> str:
    r = await client.get(
        f"{API_BASE}/generate/status/{task_id}",
        headers={"Client-Agent": CLIENT_AGENT},
        timeout=30.0,
    )
    if r.status_code != 200:
        raise HordeError(f"status failed: {r.status_code} {r.text[:200]}")
    data = r.json()
    gens = data.get("generations") or []
    if not gens:
        raise HordeError(f"no generations in status: {data}")
    url = gens[0].get("img")
    if not url:
        raise HordeError(f"no img url in generation: {gens[0]}")
    return url


async def generate_image(
    prompt: str,
    nsfw: bool | None = None,
    max_wait_s: float = 480.0,  # 8 minutes
    source_image_b64: str | None = None,
    denoising_strength: float = 0.55,
) -> bytes:
    """Generate a single image.

    If source_image_b64 is provided, img2img is used (the model starts from
    the reference image and "noises" it toward the prompt). A low
    denoising_strength (0.3-0.5) preserves the reference heavily; a high
    one (0.6-0.8) moves away from it.
    """
    is_nsfw = CONFIG.horde_nsfw if nsfw is None else bool(nsfw)
    full_prompt = f"{prompt.strip()} ### {DEFAULT_NEGATIVE}"

    params = dict(DEFAULT_PARAMS)
    if CONFIG.horde_seed:
        params["seed"] = CONFIG.horde_seed
    if source_image_b64:
        params["denoising_strength"] = max(0.1, min(0.95, denoising_strength))

    payload = {
        "prompt": full_prompt,
        "params": params,
        "nsfw": is_nsfw,
        "censor_nsfw": False,
        "trusted_workers": False,
        "slow_workers": True,
        "r2": True,
        "replacement_filter": False,
    }
    if CONFIG.horde_models:
        payload["models"] = CONFIG.horde_models
    if source_image_b64:
        payload["source_image"] = source_image_b64
        payload["source_processing"] = "img2img"

    async with httpx.AsyncClient() as client:
        task_id = await _submit(client, payload)
        await _wait_for_done(client, task_id, max_wait_s=max_wait_s)
        url = await _fetch_result_url(client, task_id)
        log.info("horde result URL: %s", url)
        r = await client.get(url, timeout=60.0)
        if r.status_code != 200:
            raise HordeError(f"image download failed: {r.status_code}")
        img_bytes = r.content
        # Detect the censorship banner by anomalous size.
        if len(img_bytes) < CENSORSHIP_SIZE_THRESHOLD:
            log.warning("image looks censored (size=%d bytes, below threshold)", len(img_bytes))
            raise CensorshipDetected(
                f"Horde returned a censorship banner (size={len(img_bytes)}B)"
            )
        return img_bytes
