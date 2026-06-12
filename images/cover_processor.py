from pathlib import Path
from PIL import Image
from loguru import logger

MAX_SIZE_BYTES = 2 * 1024 * 1024  # 2MB WeChat limit
ALLOWED_FORMATS = {".jpg", ".jpeg", ".png"}


def process_cover_image(source_path: str, output_path: str = None) -> str:
    """Validate, resize, and compress a cover image. Returns output path."""
    p = Path(source_path)
    if not p.exists():
        raise FileNotFoundError(f"封面图不存在: {source_path}")
    if p.suffix.lower() not in ALLOWED_FORMATS:
        raise ValueError(f"封面图格式不支持: {p.suffix}（仅支持 jpg/png）")

    if output_path is None:
        output_path = str(p.parent / f"cover_processed{p.suffix}")

    img = Image.open(p)

    # Convert RGBA to RGB for JPEG
    if img.mode in ("RGBA", "P") and p.suffix.lower() in {".jpg", ".jpeg"}:
        img = img.convert("RGB")

    # Resize if too large (WeChat recommends max 2048 width for covers)
    max_w = 2048
    if img.width > max_w:
        ratio = max_w / img.width
        new_size = (max_w, int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        logger.info(f"图片已缩放至 {new_size}")

    # Save with compression
    quality = 85
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() in {".jpg", ".jpeg"}:
        img.save(out, format="JPEG", quality=quality, optimize=True)
    else:
        img.save(out, format="PNG", optimize=True)

    size = out.stat().st_size
    logger.info(f"封面图处理完成: {out} ({size // 1024}KB)")

    if size > MAX_SIZE_BYTES:
        logger.warning(f"封面图仍超过 2MB（{size // 1024}KB），WeChat 可能拒绝上传")

    return str(out)


def generate_body_images(article_json: dict) -> list:
    """Placeholder for body image generation (Phase 2)."""
    prompts = article_json.get("body_image_prompts", [])
    logger.info(f"正文图片生成暂未启用（{len(prompts)} 个提示词已记录）")
    return []


def upload_body_images(images: list) -> list:
    return []


def replace_local_image_paths_with_wechat_urls(html: str, uploaded_images: list) -> str:
    return html
