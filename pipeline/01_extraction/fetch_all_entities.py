#!/usr/bin/env python3
"""
Fetch All Entities - Data Collection Script

This script fetches ALL entity data from Wikipedia (via Wikimedia REST API),
regardless of whether they have images or not. It downloads ALL images with
.png, .jpg, .jpeg extensions without filtering by role/type.

Key differences from wikipedia_processor.py:
- Saves ALL entities (even those without images)
- Downloads ALL images with supported extensions (no role filtering)
- Separates data collection from processing logic

Features:
1. Parallel list fetching
2. Parallel entity processing
3. Parallel image downloads
4. Dynamic rate limiting
5. Multi-language support
6. Saves complete entity data for future processing
"""

import os
import json
import re
import time
import yaml
import requests
from urllib.parse import urlparse, unquote, quote
from requests.adapters import HTTPAdapter, Retry
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

# =============================================================================
# CONFIGURATION
# =============================================================================

API_SESSION = None
DEFAULT_UA = 'WikipediaEntityFetcher'

# Global rate limit tracking (thread-safe)
import threading
RATE_LIMIT_LOCK = threading.Lock()
CURRENT_RATE_LIMIT = 0.8
RATE_LIMIT_ADJUSTMENTS = 0


def configure_rest_api_session(token: Optional[str] = None, lang: str = 'pt'):
    """Create a requests session configured for Wikimedia REST API."""
    global API_SESSION
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    s.mount('http://', adapter)
    s.mount('https://', adapter)

    headers = {
        'User-Agent': DEFAULT_UA,
        'Accept': 'application/json'
    }

    if token:
        headers['Authorization'] = f'Bearer {token}'
        print("✓ API token configured (5,000 requests/hour)")
    else:
        print("⚠️  No API token (500 requests/hour limit)")

    s.headers.update(headers)
    API_SESSION = s
    return s


def configure_rate_limit(has_token: bool, manual_rate: Optional[float] = None):
    """Configure rate limiting based on token availability."""
    global CURRENT_RATE_LIMIT

    if manual_rate is not None:
        CURRENT_RATE_LIMIT = manual_rate
    elif has_token:
        CURRENT_RATE_LIMIT = 0.8
    else:
        CURRENT_RATE_LIMIT = 7.5

    print(f"✓ Rate limit set to {CURRENT_RATE_LIMIT}s per request")


def adjust_rate_limit_on_429():
    """Dynamically increase rate limit when hitting 429 errors."""
    global CURRENT_RATE_LIMIT, RATE_LIMIT_ADJUSTMENTS

    with RATE_LIMIT_LOCK:
        old_rate = CURRENT_RATE_LIMIT
        CURRENT_RATE_LIMIT = min(CURRENT_RATE_LIMIT * 2, 10.0)
        RATE_LIMIT_ADJUSTMENTS += 1

        print("\n⚠️  [RATE LIMIT] 429 Error detected! Adjusting rate limit:")
        print(f"   Old: {old_rate:.3f}s → New: {CURRENT_RATE_LIMIT:.3f}s")
        print(f"   Total adjustments: {RATE_LIMIT_ADJUSTMENTS}\n")

        time.sleep(CURRENT_RATE_LIMIT * 2)


def get_current_rate_limit() -> float:
    """Get the current dynamic rate limit in a thread-safe way."""
    with RATE_LIMIT_LOCK:
        return CURRENT_RATE_LIMIT


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load settings from a YAML file; empty mapping when the file is absent."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['settings']


def sanitize_filename(name: str) -> str:
    """Remove special characters from filename."""
    sanitized = re.sub(r'[\\/*?:"<>|]', "", name)
    return sanitized.replace(" ", "_")


def extract_title_from_url(url: str) -> Optional[str]:
    """Extract Wikipedia page title from URL."""
    try:
        parsed = urlparse(url)
        path_parts = parsed.path.split('/')
        if len(path_parts) >= 3 and path_parts[1] == 'wiki':
            title = unquote(path_parts[2])
            return title
        return None
    except Exception as e:
        print(f"  [ERROR] Failed to extract title from URL '{url}': {e}")
        return None


# =============================================================================
# WIKIMEDIA REST API FUNCTIONS
# =============================================================================

def get_page_content_rest(
    page_title: str,
    lang: str = 'pt',
    rate_limit: float = 0.8
) -> Tuple[Optional[Dict], str, List[str]]:
    """
    Fetch Wikipedia page content using Wikimedia REST API.

    Returns:
        Tuple of (page data dict, final page title, categories list)
    """
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            current_limit = get_current_rate_limit()
            time.sleep(current_limit)

            encoded_title = quote(page_title, safe='')
            base_url = f"https://api.wikimedia.org/core/v1/wikipedia/{lang}"

            page_url = f"{base_url}/page/{encoded_title}"
            response = API_SESSION.get(page_url, timeout=15)
            response.raise_for_status()
            page_data = response.json()

            final_title = page_data.get('title', page_title)

            # Extract categories if available in page data
            categories = []

            return page_data, final_title, categories

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                adjust_rate_limit_on_429()
                retry_count += 1
                if retry_count < max_retries:
                    print(f"  [RETRY {retry_count}/{max_retries}] Retrying '{page_title}' with new rate limit...")
                    continue
                else:
                    print(f"  [ERROR] Max retries reached for '{page_title}'")
                    return None, page_title, []
            elif e.response.status_code == 404:
                print(f"  [ERROR] Page not found: '{page_title}'")
                return None, page_title, []
            else:
                raise

        except Exception as e:
            if '429' in str(e) or 'Too Many Requests' in str(e):
                adjust_rate_limit_on_429()
                retry_count += 1
                if retry_count < max_retries:
                    print(f"  [RETRY {retry_count}/{max_retries}] Retrying '{page_title}' with new rate limit...")
                    continue

            print(f"  [ERROR] Failed to fetch page '{page_title}': {e}")
            return None, page_title, []

    return None, page_title, []


def get_links_from_list_page_rest(
    page_title: str,
    lang: str = 'pt',
    rate_limit: float = 0.8
) -> Set[str]:
    """Extract all valid article links from a Wikipedia list page."""
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            current_limit = get_current_rate_limit()
            time.sleep(current_limit)

            encoded_title = quote(page_title, safe='')
            base_url = f"https://api.wikimedia.org/core/v1/wikipedia/{lang}"
            page_url = f"{base_url}/page/{encoded_title}"

            response = API_SESSION.get(page_url, timeout=15)
            response.raise_for_status()
            page_data = response.json()

            source = page_data.get('source', '')
            if not source:
                print(f"  [WARNING] No wikitext source found for '{page_title}'")
                return set()

            all_links = set()
            link_pattern = re.compile(r'\[\[([^\[\]|#]+?)(?:\|[^\[\]]*?)?\]\]')

            for match in link_pattern.finditer(source):
                link_title = match.group(1).strip()

                if not link_title:
                    continue

                if link_title.startswith(':'):
                    link_title = link_title[1:]

                # Skip "Lista" pages
                if 'lista' in link_title.lower() or 'list of' in link_title.lower():
                    continue

                # Skip interwiki links
                interwiki_prefixes = [
                    'd:', 'wikidata:', 'commons:', 'c:',
                    'en:', 'es:', 'fr:', 'de:', 'it:', 'pt:', 'ru:', 'ja:', 'zh:', 'ar:',
                ]
                skip_interwiki = False
                for prefix in interwiki_prefixes:
                    if link_title.lower().startswith(prefix):
                        skip_interwiki = True
                        break
                if skip_interwiki:
                    continue

                # Skip namespace pages
                namespace_patterns = [
                    'ficheiro:', 'file:', 'imagem:', 'image:',
                    'especial:', 'special:', 'wikipedia:', 'wikipédia:',
                    'ajuda:', 'help:', 'portal:', 'predefinição:', 'template:',
                    'discussão:', 'talk:', 'usuário:', 'user:',
                    'módulo:', 'module:', 'mediawiki:',
                ]
                skip_namespace = False
                for pattern in namespace_patterns:
                    if link_title.lower().startswith(pattern):
                        skip_namespace = True
                        break
                if skip_namespace:
                    continue

                page_path = link_title.replace(' ', '_')
                absolute_url = f"https://{lang}.wikipedia.org/wiki/{page_path}"
                all_links.add(absolute_url)

            return all_links

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                adjust_rate_limit_on_429()
                retry_count += 1
                if retry_count < max_retries:
                    print(f"  [RETRY {retry_count}/{max_retries}] Retrying list '{page_title}' with new rate limit...")
                    continue
                else:
                    print(f"  [ERROR] Max retries reached for list '{page_title}'")
                    return set()
            elif e.response.status_code == 404:
                print(f"  [ERROR] Page not found: '{page_title}'")
                return set()
            else:
                print(f"  [ERROR] HTTP {e.response.status_code}: {e}")
                raise

        except Exception as e:
            if '429' in str(e) or 'Too Many Requests' in str(e):
                adjust_rate_limit_on_429()
                retry_count += 1
                if retry_count < max_retries:
                    print(f"  [RETRY {retry_count}/{max_retries}] Retrying list '{page_title}' with new rate limit...")
                    continue

            print(f"  [ERROR] Failed to fetch list page '{page_title}': {e}")
            return set()

    return set()


def clean_references(text: str) -> str:
    """
    Remove Wikipedia reference tags (<ref>...</ref>) from text.
    """
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<ref[^>]*?/>', '', text, flags=re.IGNORECASE)
    return text.strip()


def extract_images_from_wikitext_section(wikitext: str, lang: str = 'pt') -> Tuple[List[Dict], str]:
    """
    Extract image information (filename and caption) from wikitext and remove the image markup from text.
    Also removes template transclusions {{...}}.
    """
    images = []
    image_prefixes = ['ficheiro:', 'file:', 'imagem:', 'image:']

    # Collect all image positions and data
    image_ranges = []
    i = 0
    while i < len(wikitext):
        if wikitext[i:i+2] == '[[':
            rest = wikitext[i+2:].lower()
            is_image = False
            prefix_len = 0

            for prefix in image_prefixes:
                if rest.startswith(prefix):
                    is_image = True
                    prefix_len = len(prefix)
                    break

            if is_image:
                start = i
                depth = 1
                j = i + 2

                while j < len(wikitext) - 1 and depth > 0:
                    if wikitext[j:j+2] == '[[':
                        depth += 1
                        j += 2
                    elif wikitext[j:j+2] == ']]':
                        depth -= 1
                        j += 2
                    else:
                        j += 1

                if depth == 0:
                    end = j
                    full_markup = wikitext[start:end]
                    inner = full_markup[2:-2]

                    # Split by | carefully handling nested brackets
                    parts = []
                    current_part = []
                    bracket_depth = 0
                    k = 0

                    while k < len(inner):
                        if k < len(inner) - 1 and inner[k:k+2] == '[[':
                            bracket_depth += 1
                            current_part.append('[[')
                            k += 2
                        elif k < len(inner) - 1 and inner[k:k+2] == ']]':
                            bracket_depth -= 1
                            current_part.append(']]')
                            k += 2
                        elif inner[k] == '|' and bracket_depth == 0:
                            parts.append(''.join(current_part))
                            current_part = []
                            k += 1
                        else:
                            current_part.append(inner[k])
                            k += 1

                    if current_part:
                        parts.append(''.join(current_part))

                    filename = parts[0][prefix_len:].strip() if len(parts) > 0 else ''

                    # Extract caption
                    caption = ''
                    if len(parts) > 1:
                        options = ['thumb', 'thumbnail', 'frame', 'miniaturadaimagem', 'miniatura',
                                   'left', 'right', 'center', 'centre', 'none',
                                   'border', 'frameless', 'sem_moldura']

                        for part in reversed(parts[1:]):
                            part_lower = part.strip().lower()
                            if 'px' in part_lower or part_lower in options:
                                continue
                            if part_lower.startswith(('link=', 'alt=')):
                                continue
                            caption = part.strip()
                            break

                    # Only accept PNG, JPG, JPEG files
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        cleaned_caption = caption
                        if cleaned_caption:
                            wiki_link_pattern = re.compile(r'\[\[([^\]|]+?)(?:\|([^\]]+?))?\]\]')
                            cleaned_caption = wiki_link_pattern.sub(lambda m: m.group(2) if m.group(2) else m.group(1), cleaned_caption)
                            cleaned_caption = clean_references(cleaned_caption)

                        images.append({
                            'filename': filename,
                            'caption': cleaned_caption
                        })

                    image_ranges.append((start, end))
                    i = end
                    continue

        i += 1

    # Build cleaned text
    if image_ranges:
        cleaned_parts = []
        last_end = 0

        for start, end in image_ranges:
            cleaned_parts.append(wikitext[last_end:start])
            last_end = end

        cleaned_parts.append(wikitext[last_end:])
        cleaned_text = ''.join(cleaned_parts)
    else:
        cleaned_text = wikitext

    # Remove template transclusions: {{...}}
    while '{{' in cleaned_text:
        start = cleaned_text.find('{{')
        if start == -1:
            break

        depth = 0
        end = -1
        for i in range(start, len(cleaned_text) - 1):
            if cleaned_text[i:i+2] == '{{':
                depth += 1
                continue
            elif cleaned_text[i:i+2] == '}}':
                depth -= 1
                if depth == 0:
                    end = i + 2
                    break

        if end != -1:
            cleaned_text = cleaned_text[:start] + cleaned_text[end:]
        else:
            cleaned_text = cleaned_text[:start] + cleaned_text[start+2:]

    return images, cleaned_text.strip()


def extract_all_images_from_wikitext(wikitext: str) -> List[str]:
    """
    Extract ALL image filenames from wikitext, regardless of role/type.
    Only filters by extension (.png, .jpg, .jpeg).
    """
    images = []

    # Pattern to match image links
    # Matches [[Ficheiro:...]], [[File:...]], [[Imagem:...]], [[Image:...]]
    file_pattern = re.compile(
        r'\[\[(?:ficheiro|file|imagem|image):([^\]|]+?)(?:\|[^\]]*?)?\]\]',
        re.IGNORECASE
    )

    for match in file_pattern.finditer(wikitext):
        filename = match.group(1).strip()

        # Only accept PNG, JPG, JPEG files
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            images.append(filename)

    return images


def extract_content_from_page_source(page_data: Dict, lang: str = 'pt') -> List[Dict]:
    """
    Extract content from Wikipedia page source (wikitext).
    Parses sections from wikitext markers (== Section ==).
    Returns cleaned content organized by sections.
    """
    sections_data = []
    # Language-specific ignored sections
    ignored_sections_by_lang = {
        'pt': {'Bibliografia', 'Referências', 'Ligações externas', 'Ver também', 'Notas'},
        'en': {'Bibliography', 'References', 'External links', 'See also', 'Notes'},
        'es': {'Bibliografía', 'Referencias', 'Enlaces externos', 'Véase también', 'Notas'},
    }
    IGNORED_SECTIONS = ignored_sections_by_lang.get(lang, ignored_sections_by_lang['pt'])

    # Get source (wikitext)
    source = page_data.get('source', '')

    if not source:
        return sections_data

    # Parse sections from wikitext format: == Section Name ==
    section_pattern = re.compile(r'\n(={2,4})\s*(.+?)\s*\1\s*\n')

    # Find all section headings
    matches = list(section_pattern.finditer(source))

    if matches:
        # Extract lead section (before first heading)
        lead_content = source[:matches[0].start()].strip()
        if lead_content:
            lead_images_info, lead_content_cleaned = extract_images_from_wikitext_section(lead_content, lang)
            lead_content_cleaned = clean_references(lead_content_cleaned)
            if lead_content_cleaned:
                sections_data.append({
                    'heading': 'Lead Section',
                    'content': [lead_content_cleaned],
                    'images': lead_images_info
                })

        # Extract each section
        for i, match in enumerate(matches):
            section_title = match.group(2).strip()
            section_title = clean_references(section_title)

            # Skip ignored sections
            if section_title in IGNORED_SECTIONS:
                continue

            # Get content between this heading and the next
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(source)
            section_content = source[start_pos:end_pos].strip()

            # Extract images and clean content
            section_images_info, section_content_cleaned = extract_images_from_wikitext_section(section_content, lang)
            section_content_cleaned = clean_references(section_content_cleaned)

            if section_content_cleaned or section_images_info:
                sections_data.append({
                    'heading': section_title,
                    'content': [section_content_cleaned] if section_content_cleaned else [],
                    'images': section_images_info
                })
    else:
        # No sections found, create single section with all content
        main_images_info, clean_content = extract_images_from_wikitext_section(source, lang)
        clean_content = clean_references(clean_content)
        if clean_content or main_images_info:
            sections_data.append({
                'heading': 'Main Content',
                'content': [clean_content] if clean_content else [],
                'images': main_images_info
            })

    return sections_data


def get_file_metadata_rest(
    filename: str,
    lang: str = 'pt',
    rate_limit: float = 0.8
) -> Optional[Dict]:
    """Get detailed file metadata from Commons or local Wikipedia."""
    try:
        current_limit = get_current_rate_limit()
        time.sleep(current_limit)

        # Clean filename
        if ':' in filename:
            clean_name = filename.split(':', 1)[1]
        else:
            clean_name = filename

        encoded_name = quote(clean_name, safe='')

        # Try Commons first
        file_url = f"https://api.wikimedia.org/core/v1/commons/file/File:{encoded_name}"

        response = API_SESSION.get(file_url, timeout=15)
        response.raise_for_status()
        data = response.json()

        file_ext = None
        if '.' in clean_name:
            file_ext = clean_name.rsplit('.', 1)[1].lower()

        preferred = data.get('preferred', {})
        original = data.get('original', {})

        return {
            'source_url': preferred.get('url') or original.get('url'),
            'width': preferred.get('width') or original.get('width'),
            'height': preferred.get('height') or original.get('height'),
            'mime': file_ext,
            'size_bytes': original.get('size'),
            'original_url': original.get('url'),
            'original_filename': filename
        }

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            # Try local Wikipedia
            return get_file_metadata_local_wiki(filename, lang, rate_limit)
        else:
            print(f"  [WARNING] Failed to get file metadata for '{filename}': {e}")
            return None

    except Exception as e:
        print(f"  [WARNING] Failed to get file metadata for '{filename}': {e}")
        return None


def get_file_metadata_local_wiki(
    filename: str,
    lang: str = 'pt',
    rate_limit: float = 0.8
) -> Optional[Dict]:
    """Get file metadata from local Wikipedia using MediaWiki Action API."""
    try:
        current_limit = get_current_rate_limit()
        time.sleep(current_limit)

        if ':' in filename:
            clean_name = filename.split(':', 1)[1]
        else:
            clean_name = filename

        api_url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'titles': f'File:{clean_name}',
            'prop': 'imageinfo',
            'iiprop': 'url|size|dimensions|mime',
            'format': 'json'
        }

        response = API_SESSION.get(api_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        pages = data.get('query', {}).get('pages', {})
        if not pages:
            return None

        page = list(pages.values())[0]

        if int(page.get('pageid', -1)) < 0:
            return None

        imageinfo = page.get('imageinfo', [])
        if not imageinfo:
            return None

        info = imageinfo[0]

        file_ext = None
        if '.' in clean_name:
            file_ext = clean_name.rsplit('.', 1)[1].lower()

        return {
            'source_url': info.get('url'),
            'width': info.get('width'),
            'height': info.get('height'),
            'mime': file_ext,
            'size_bytes': info.get('size'),
            'original_url': info.get('url'),
            'original_filename': filename
        }

    except Exception as e:
        print(f"  [WARNING] Failed to get local file metadata for '{filename}': {e}")
        return None


def download_image_from_url(image_url: str) -> Optional[bytes]:
    """Download image from a direct URL (CDN download - no rate limiting)."""
    try:
        img_response = API_SESSION.get(image_url, timeout=30)
        img_response.raise_for_status()
        return img_response.content
    except Exception as e:
        print(f"  [WARNING] Failed to download image from '{image_url}': {e}")
        return None


def download_images_parallel(
    image_metadata_list: List[Dict],
    configured_max_workers: int = 50
) -> List[Tuple[Dict, Optional[bytes]]]:
    """Download multiple images in parallel."""
    results = []
    num_images = len(image_metadata_list)
    if num_images == 0:
        return []

    actual_max_workers = min(num_images, configured_max_workers)

    with ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
        future_to_metadata = {
            executor.submit(download_image_from_url, metadata['source_url']): metadata
            for metadata in image_metadata_list
        }

        for future in as_completed(future_to_metadata):
            metadata = future_to_metadata[future]
            try:
                image_bytes = future.result()
                results.append((metadata, image_bytes))
            except Exception as e:
                print(f"  [WARNING] Exception downloading {metadata.get('source_url')}: {e}")
                results.append((metadata, None))

    return results


# =============================================================================
# FILTERING FUNCTIONS
# =============================================================================

def is_category_page(url: str) -> bool:
    """Check if URL is a category page."""
    return '/wiki/Categoria:' in url or '/wiki/Category:' in url


def is_date_related_page(title: str, lang: str = 'pt') -> bool:
    """
    Check if page title is date-related (years, day pages).

    Filters out:
    - Titles that are only numbers (years: "1999", "2023")
    - Titles with day-month patterns (e.g., "12 de Janeiro")
    """
    months_by_lang = {
        'pt': ['janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho',
               'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro'],
        'en': ['january', 'february', 'march', 'april', 'may', 'june',
               'july', 'august', 'september', 'october', 'november', 'december'],
    }

    months = [m.lower() for m in months_by_lang.get(lang, [])]

    # Check if title is only numbers (years)
    if re.match(r'^\d+$', title.strip()):
        return True

    # Check for month patterns
    for month in months:
        if f'de {month}' in title.lower():
            return True
        if re.search(fr'{month}\s+\d+', title.lower()):
            return True

    return False


def entity_already_processed(entity_title: str, entities_dir: str) -> bool:
    """Check if entity has already been processed."""
    entity_dir = os.path.join(entities_dir, sanitize_filename(entity_title))
    entity_data_path = os.path.join(entity_dir, "entity_data.json")
    return os.path.exists(entity_data_path)


# =============================================================================
# ENTITY PROCESSING
# =============================================================================

def fetch_entity_data(
    entity_title: str,
    output_dir: str,
    entity_url: str,
    lang: str = 'pt',
    rate_limit: float = 0.8,
    parallel_downloads: bool = True,
    download_workers: int = 50,
    quiet: bool = False
) -> bool:
    """
    Fetch and save ALL data for a single entity, regardless of images.
    Downloads ALL images with .png, .jpg, .jpeg extensions (no role filtering).

    Returns:
        True if successful, False if failed
    """
    if not quiet:
        print(f"  Fetching: {entity_title}")

    # Fetch page content
    page_data, final_title, categories = get_page_content_rest(
        entity_title,
        lang=lang,
        rate_limit=rate_limit
    )

    if page_data is None:
        if not quiet:
            print("    [FAILED] Could not retrieve page")
        return False

    # Check for redirects
    if final_title != entity_title:
        if not quiet:
            print(f"    [REDIRECT] {entity_title} → {final_title}")
        entity_title = final_title

        # Skip if already processed
        entities_base_dir = os.path.join(output_dir, "entities")
        if entity_already_processed(entity_title, entities_base_dir):
            if not quiet:
                print(f"    [SKIP] Already processed as '{entity_title}'")
            return True

    # Extract wikitext
    wikitext = page_data.get('source', '')

    # Extract processed sections with cleaned text
    sections = extract_content_from_page_source(page_data, lang)

    # Extract ALL images from wikitext (no role filtering)
    all_image_filenames = extract_all_images_from_wikitext(wikitext)

    # Create entity directory
    entity_dir = os.path.join(output_dir, sanitize_filename(entity_title))
    os.makedirs(entity_dir, exist_ok=True)

    if not quiet and all_image_filenames:
        print(f"    Found {len(all_image_filenames)} images")

    # Fetch metadata for all images
    metadata_list = []
    for filename in all_image_filenames:
        metadata = get_file_metadata_rest(filename, lang, rate_limit)
        if metadata:
            metadata_list.append(metadata)

    # Download images
    images_data = []
    if metadata_list:
        if parallel_downloads:
            download_results = download_images_parallel(metadata_list, configured_max_workers=download_workers)
        else:
            download_results = []
            for metadata in metadata_list:
                image_bytes = download_image_from_url(metadata['source_url'])
                download_results.append((metadata, image_bytes))

        # Save all downloaded images
        for metadata, image_bytes in download_results:
            if not image_bytes:
                continue

            # Determine filename
            if 'original_filename' in metadata and ':' in metadata['original_filename']:
                fname = metadata['original_filename'].split(":", 1)[1]
            else:
                fname = metadata.get('original_filename', 'image')

            base_filename = sanitize_filename(os.path.splitext(fname)[0])
            ext = os.path.splitext(fname)[1] or '.jpg'

            image_path = os.path.join(entity_dir, f"{base_filename}{ext}")
            meta_path = os.path.join(entity_dir, f"{base_filename}.json")

            # Save image
            with open(image_path, 'wb') as f:
                f.write(image_bytes)

            # Save metadata
            metadata_to_save = {
                'source_url': metadata['source_url'],
                'width': metadata.get('width'),
                'height': metadata.get('height'),
                'mime': metadata.get('mime'),
                'size_bytes': metadata.get('size_bytes'),
                'original_filename': metadata.get('original_filename', fname)
            }

            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_to_save, f, ensure_ascii=False, indent=2)

            images_data.append({
                'original_filename': metadata.get('original_filename', fname),
                'local_filename': os.path.basename(image_path)
            })

    # Save entity data (even if no images)
    entity_data = {
        'title': final_title,
        'url': entity_url,
        'language': lang,
        'categories': categories,
        'wikitext': wikitext,  # Raw wikitext from Wikimedia
        'sections': sections,  # Processed, clean sections with text
        'has_images': len(images_data) > 0,
        'images': images_data,
        'total_images': len(images_data)
    }

    entity_data_path = os.path.join(entity_dir, "entity_data.json")
    with open(entity_data_path, 'w', encoding='utf-8') as f:
        json.dump(entity_data, f, ensure_ascii=False, indent=2)

    if not quiet:
        print(f"    ✓ Saved with {len(images_data)} images")

    return True


# =============================================================================
# PARALLEL PROCESSING
# =============================================================================

def process_entity_batch(args):
    """Worker function for multiprocessing."""
    worker_id, entities_batch, entities_dir, lang, rate_limit, api_token, parallel_downloads, download_workers = args

    # Initialize session for this worker
    configure_rest_api_session(api_token, lang)
    configure_rate_limit(bool(api_token), rate_limit)

    processed = []
    failed = []

    for entity in tqdm(entities_batch,
                      desc=f"Worker {worker_id}",
                      unit="entity",
                      position=worker_id,
                      leave=True):
        success = fetch_entity_data(
            entity['title'],
            entities_dir,
            entity['url'],
            lang,
            rate_limit,
            parallel_downloads,
            download_workers,
            quiet=True
        )

        if success:
            processed.append(entity)
        else:
            failed.append(entity)

    return {
        'worker_id': worker_id,
        'processed': processed,
        'failed': failed
    }


def fetch_single_list(args):
    """Worker function to fetch a single list page."""
    list_url, lang, rate_limit = args

    page_title = extract_title_from_url(list_url)
    if not page_title:
        return (list_url, set())

    article_links = get_links_from_list_page_rest(page_title, lang=lang, rate_limit=rate_limit)
    return (list_url, article_links)


def collect_all_links_from_lists(
    list_urls: List[str],
    lang: str = 'pt',
    rate_limit: float = 0.8,
    max_workers: int = 5
) -> Dict[str, Set[str]]:
    """Collect all unique article links from all list pages."""
    print(f"\n{'='*80}")
    print("PHASE 1: Collecting all links from lists")
    print(f"{'='*80}\n")
    print(f"Processing {len(list_urls)} lists with {max_workers} parallel workers\n")

    all_links = {}
    fetch_args = [(url, lang, rate_limit) for url in list_urls]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_single_list, args): args[0] for args in fetch_args}

        completed = 0
        for future in as_completed(future_to_url):
            list_url = future_to_url[future]
            completed += 1
            try:
                list_url, article_links = future.result()
                all_links[list_url] = article_links
                print(f"[{completed}/{len(list_urls)}] {list_url}: {len(article_links)} links")
            except Exception as e:
                print(f"[{completed}/{len(list_urls)}] [ERROR] {list_url}: {e}")
                all_links[list_url] = set()

    print(f"\nCompleted fetching {len(list_urls)} lists")
    return all_links


def analyze_entities(
    all_links: Dict[str, Set[str]],
    entities_dir: str,
    output_dir: str,
    lang: str = 'pt'
) -> Dict:
    """Analyze all collected links to determine what will be processed/skipped."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(entities_dir, exist_ok=True)

    all_unique_urls = set()
    for links in all_links.values():
        all_unique_urls.update(links)

    print(f"\n{'='*80}")
    print("PHASE 2: Analyzing entities")
    print(f"{'='*80}\n")

    total_collected = len(all_unique_urls)
    category_pages = []
    date_filtered = []
    already_processed = []
    to_process = []

    print(f"📊 Total unique entities collected: {total_collected}")

    for url in sorted(all_unique_urls):
        # Filter category pages
        if is_category_page(url):
            parsed_url = urlparse(url)
            path_parts = parsed_url.path.split('/')
            page_title = unquote(path_parts[2]) if len(path_parts) >= 3 else url
            category_pages.append({'url': url, 'title': page_title})
            continue

        parsed_url = urlparse(url)
        path_parts = parsed_url.path.split('/')
        if len(path_parts) < 3 or path_parts[1] != 'wiki':
            continue

        page_title = unquote(path_parts[2])

        # Filter date-related pages
        if is_date_related_page(page_title, lang):
            date_filtered.append({'url': url, 'title': page_title})
        # Check if already processed
        elif entity_already_processed(page_title, entities_dir):
            already_processed.append({'url': url, 'title': page_title})
        else:
            to_process.append({'url': url, 'title': page_title})

    # Print statistics
    after_category_filter = total_collected - len(category_pages)
    after_date_filter = after_category_filter - len(date_filtered)
    after_already_processed_filter = after_date_filter - len(already_processed)

    print(f"📊 After category filtering: {after_category_filter} entities (removed {len(category_pages)})")
    print(f"📊 After date filtering: {after_date_filter} entities (removed {len(date_filtered)})")
    print(f"📊 After already-processed filtering: {after_already_processed_filter} entities (skipped {len(already_processed)})")
    print(f"📊 To process: {len(to_process)} entities")
    print()

    # Save filtered lists
    if category_pages:
        categories_file = os.path.join(output_dir, "categories.txt")
        with open(categories_file, 'w', encoding='utf-8') as f:
            for item in category_pages:
                f.write(f"{item['url']}\n")
        print(f"✓ Category pages saved to: {categories_file}")

    if date_filtered:
        date_filtered_file = os.path.join(output_dir, "date_filtered.txt")
        with open(date_filtered_file, 'w', encoding='utf-8') as f:
            for item in date_filtered:
                f.write(f"{item['url']}\n")
        print(f"✓ Date-filtered pages saved to: {date_filtered_file}")

    print()

    return {
        'total_collected': total_collected,
        'category_pages': category_pages,
        'date_filtered': date_filtered,
        'already_processed': already_processed,
        'to_process': to_process
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    config = load_config()

    output_dir = config.get('output_dir', str(DATA / '01_extraction' / 'newdata' / 'temporal_relevant'))
    lang = config.get('language', 'pt')
    lists_file = config.get('lists_file', 'lists.txt')
    rate_limit = config.get('rate_limit_seconds', 0.8)
    num_workers = config.get('num_workers', 4)
    list_fetch_workers = config.get('list_fetch_workers', 5)
    parallel_downloads = config.get('parallel_image_downloads', True)
    download_workers = config.get('image_download_workers', 50)
    api_token = config.get('wikimedia_api_token')

    # Initialize REST API session
    configure_rest_api_session(api_token, lang)
    configure_rate_limit(bool(api_token), rate_limit)

    print("="*80)
    print("Fetch All Entities - Data Collection Script")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Language: {lang}")
    print(f"Lists file: {lists_file}")
    print(f"Rate limit: {rate_limit}s")
    print(f"Entity workers: {num_workers}")
    print(f"List fetch workers: {list_fetch_workers}")
    print(f"Parallel image downloads: {parallel_downloads}")
    print(f"Image download workers: {download_workers}")

    # Read list URLs
    if not os.path.exists(lists_file):
        print(f"\n[ERROR] Lists file not found: {lists_file}")
        return

    with open(lists_file, 'r', encoding='utf-8') as f:
        list_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if not list_urls:
        print("\n[ERROR] No list URLs found")
        return

    print(f"\nFound {len(list_urls)} list(s) to process\n")

    entities_dir = os.path.join(output_dir, "entities")
    os.makedirs(entities_dir, exist_ok=True)

    # PHASE 1: Collect all links
    all_links = collect_all_links_from_lists(list_urls, lang, rate_limit, list_fetch_workers)

    # PHASE 2: Analyze entities
    analysis = analyze_entities(all_links, entities_dir, output_dir, lang)

    # PHASE 3: Fetch all entity data
    print(f"\n{'='*80}")
    print("PHASE 3: Fetching entity data")
    print(f"{'='*80}\n")

    processed_entities = []
    failed_entities = []

    if analysis['to_process']:
        print(f"Fetching {len(analysis['to_process'])} entities...\n")
        print(f"Using {num_workers} parallel workers\n")

        # Split entities into batches
        entities_list = analysis['to_process']
        batch_size = len(entities_list) // num_workers + (1 if len(entities_list) % num_workers else 0)
        batches = []

        for i in range(num_workers):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(entities_list))
            if start_idx < len(entities_list):
                batch = entities_list[start_idx:end_idx]
                batches.append((i, batch, entities_dir, lang, rate_limit, api_token, parallel_downloads, download_workers))

        print(f"Split {len(entities_list)} entities into {len(batches)} batches\n")

        # Process batches in parallel
        with Pool(processes=num_workers) as pool:
            results = list(pool.map(process_entity_batch, batches))

        # Aggregate results
        for result in results:
            processed_entities.extend(result['processed'])
            failed_entities.extend(result['failed'])

        print("\n✓ All workers completed\n")

    # Final statistics
    print(f"\n{'='*80}")
    print("FINAL STATISTICS")
    print(f"{'='*80}")
    print(f"📊 Total unique entities collected: {analysis['total_collected']}")
    print(f"📊 Category pages (filtered): {len(analysis['category_pages'])}")
    print(f"📊 Date-related pages (filtered): {len(analysis['date_filtered'])}")
    print(f"📊 Already processed (skipped): {len(analysis['already_processed'])}")
    print(f"📊 Newly processed: {len(processed_entities)}")
    print(f"📊 Failed: {len(failed_entities)}")
    print(f"{'='*80}")

    # Save statistics
    stats_data = {
        'total_collected': analysis['total_collected'],
        'category_pages_count': len(analysis['category_pages']),
        'date_filtered_count': len(analysis['date_filtered']),
        'already_processed_count': len(analysis['already_processed']),
        'newly_processed': len(processed_entities),
        'failed': len(failed_entities),
        'processed_list': [item['title'] for item in processed_entities],
        'failed_list': [item['title'] for item in failed_entities]
    }

    stats_file = os.path.join(output_dir, "fetch_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Statistics saved to: {stats_file}")

    if RATE_LIMIT_ADJUSTMENTS > 0:
        print(f"\n⚠️  Rate limit was adjusted {RATE_LIMIT_ADJUSTMENTS} time(s) due to 429 errors")
        print(f"   Final rate limit: {CURRENT_RATE_LIMIT:.3f}s per request")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()