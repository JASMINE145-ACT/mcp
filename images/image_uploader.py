from loguru import logger
from images.cover_processor import process_cover_image
from wechat.material import upload_permanent_image


def prepare_and_upload_cover(cover_path: str, task_dir: str) -> str:
    """Process the cover image and upload it to WeChat. Returns thumb_media_id."""
    import os
    processed_path = os.path.join(task_dir, "cover_processed.jpg")
    processed = process_cover_image(cover_path, processed_path)
    logger.info(f"封面图处理完成: {processed}")
    thumb_media_id = upload_permanent_image(processed)
    return thumb_media_id
