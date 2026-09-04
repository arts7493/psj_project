from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageOps

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp')


def _normalize_name(value: str) -> str:
    text = str(value or '').strip().lower()
    return re.sub(r'[^0-9a-z가-힣]+', '', text)


def _line_code(line_no: Any) -> str:
    try:
        return f"p{int(line_no):03d}"
    except Exception:
        return ''


def _build_file_index(image_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not image_dir.exists():
        return index

    for path in image_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        index.setdefault(path.name.lower(), path)
        index.setdefault(path.stem.lower(), path)
        index.setdefault(_normalize_name(path.stem), path)
    return index


def _find_image_path(row: pd.Series, file_index: dict[str, Path]) -> str:
    explicit = str(row.get('이미지') or '').strip()
    if explicit:
        keys = [explicit.lower(), Path(explicit).name.lower(), Path(explicit).stem.lower(), _normalize_name(Path(explicit).stem)]
        for key in keys:
            if key in file_index:
                return str(file_index[key])

    name = str(row.get('이름') or '').strip()
    if name:
        keys = [name.lower(), Path(name).stem.lower(), _normalize_name(name)]
        for key in keys:
            if key in file_index:
                return str(file_index[key])

    line_code = _line_code(row.get('_csv_line_no'))
    if line_code:
        keys = [line_code.lower(), f'{line_code.lower()}.jpg', f'{line_code.lower()}.jpeg', f'{line_code.lower()}.png', f'{line_code.lower()}.webp']
        for key in keys:
            if key in file_index:
                return str(file_index[key])

    return ''


def attach_preview_images(df: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    result = df.copy()
    file_index = _build_file_index(image_dir)
    result['_preview_path'] = result.apply(lambda row: _find_image_path(row, file_index), axis=1)
    return result


def preview_image_data_uri(path_text: str, width: int = 192, height: int = 160) -> str:
    path = Path(str(path_text or '').strip())
    if not path.exists() or not path.is_file():
        return ''

    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            thumb = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            thumb.save(buffer, format='JPEG', quality=86, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
            return f'data:image/jpeg;base64,{encoded}'
    except Exception:
        return ''
