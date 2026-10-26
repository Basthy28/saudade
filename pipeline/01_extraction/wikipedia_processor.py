#!/usr/bin/env python3
"""
Wikipedia List Processor - Wikimedia REST API Version

This version uses the Wikimedia Core REST API exclusively for all data fetching.

Features:
1. Read Wikipedia list URLs from file
2. Extract ALL article links from each list (excluding navigation, "Lista", etc.)
3. For each article:
   - Check if it has images
   - Download images and extract content by section
   - Save organized entity data
4. Enhanced statistics tracking at each filtering step

API Documentation:
- Base API: https://api.wikimedia.org/wiki/Core_REST_API
- Rate limits: 5,000/hour (with token), 500/hour (without token)
"""

import os
import json
import re
import time
from pathlib import Path

import yaml
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote, quote
from requests.adapters import HTTPAdapter, Retry
from typing import List, Dict, Optional, Tuple, Set
from tqdm import tqdm
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

# =============================================================================
# CONFIGURATION
# =============================================================================

API_SESSION = None
DEFAULT_UA = 'WikipediaListProcessor'

# Global rate limit tracking (thread-safe)
import threading
RATE_LIMIT_LOCK = threading.Lock()
CURRENT_RATE_LIMIT = 0.8  # Will be updated based on token availability
RATE_LIMIT_ADJUSTMENTS = 0  # Track how many times we've adjusted

# Language-specific keyword configuration
LANG_CONFIG = {
    'pt': {
        'image_prefixes': ['ficheiro:', 'file:', 'imagem:', 'image:'],
        'infobox_image_param': ['imagem', 'logo', 'bandeira'],  # Accept 'imagem', 'logo', or 'bandeira'
        'infobox_caption_param': 'imagem_legenda',
        'ignored_sections': {'Bibliografia', 'Referências', 'Ligações externas', 'Ver também', 'Notas'},
        'category_prefix': 'Categoria'
    },
    'fr': {
        'image_prefixes': ['fichier:', 'file:', 'image:'],
        'infobox_image_param': 'image',
        'infobox_caption_param': 'légende',
        'ignored_sections': {'Bibliographie', 'Références', 'Liens externes', 'Voir aussi', 'Notes'},
        'category_prefix': 'Catégorie'
    },
    'it': {
        'image_prefixes': ['file:', 'immagine:'],
        'infobox_image_param': 'immagine',
        'infobox_caption_param': 'didascalia',
        'ignored_sections': {'Bibliografia', 'Riferimenti', 'Collegamenti esterni', 'Voci correlate', 'Note'},
        'category_prefix': 'Categoria'
    },
    'en': {
        'image_prefixes': ['file:', 'image:'],
        'infobox_image_param': 'image',
        'infobox_caption_param': 'caption',
        'ignored_sections': {'Bibliography', 'References', 'External links', 'See also', 'Notes'},
        'category_prefix': 'Category'
    },
    'de': {
        'image_prefixes': ['datei:', 'file:', 'image:', 'bild:'],
        'infobox_image_param': 'bild',
        'infobox_caption_param': 'bildunterschrift',
        'ignored_sections': {'Literatur', 'Einzelnachweise', 'Weblinks', 'Siehe auch', 'Anmerkungen'},
        'category_prefix': 'Kategorie'
    },
    'es': {
        'image_prefixes': ['archivo:', 'file:', 'image:', 'imagen:'],
        'infobox_image_param': 'imagen',
        'infobox_caption_param': 'pie_de_imagen',
        'ignored_sections': {'Bibliografía', 'Referencias', 'Enlaces externos', 'Véase también', 'Notas'},
        'category_prefix': 'Categoría'
    },
    # Add other languages as needed
}

def get_lang_config(lang: str) -> Dict:
    """Get language-specific keywords, defaulting to English."""
    return LANG_CONFIG.get(lang, LANG_CONFIG['en'])


def configure_rest_api_session(token: Optional[str] = None, lang: str = 'pt'):
    """
    Create a requests session configured for Wikimedia REST API.

    Args:
        token: Optional API token from Special:BotPasswords
        lang: Language code (pt, en, etc.)

    Returns:
        Configured session object
    """
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
    """
    Configure rate limiting based on token availability.

    Args:
        has_token: Whether an API token is configured
        manual_rate: Optional manual rate limit override
    """
    global CURRENT_RATE_LIMIT

    if manual_rate is not None:
        CURRENT_RATE_LIMIT = manual_rate
    elif has_token:
        CURRENT_RATE_LIMIT = 0.8  # 5000/hour = 0.72s + buffer
    else:
        CURRENT_RATE_LIMIT = 7.5  # 500/hour = 7.2s + buffer

    print(f"✓ Rate limit set to {CURRENT_RATE_LIMIT}s per request")


def adjust_rate_limit_on_429():
    """
    Dynamically increase rate limit when hitting 429 errors.
    Uses exponential backoff to find safe rate.
    """
    global CURRENT_RATE_LIMIT, RATE_LIMIT_ADJUSTMENTS

    with RATE_LIMIT_LOCK:
        old_rate = CURRENT_RATE_LIMIT
        # Double the rate limit (exponential backoff)
        CURRENT_RATE_LIMIT = min(CURRENT_RATE_LIMIT * 2, 10.0)  # Cap at 10 seconds
        RATE_LIMIT_ADJUSTMENTS += 1

        print("\n⚠️  [RATE LIMIT] 429 Error detected! Adjusting rate limit:")
        print(f"   Old: {old_rate:.3f}s → New: {CURRENT_RATE_LIMIT:.3f}s")
        print(f"   Total adjustments: {RATE_LIMIT_ADJUSTMENTS}\n")

        # Sleep extra time after adjustment
        time.sleep(CURRENT_RATE_LIMIT * 2)


def get_current_rate_limit() -> float:
    """Get the current dynamic rate limit in a thread-safe way."""
    with RATE_LIMIT_LOCK:
        return CURRENT_RATE_LIMIT


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['settings']


def sanitize_filename(name: str) -> str:
    """Remove special characters from filename."""
    sanitized = re.sub(r'[\\/*?:"<>|]', "", name)
    return sanitized.replace(" ", "_")


def extract_title_from_url(url: str) -> Optional[str]:
    """
    Extract Wikipedia page title from URL.

    Args:
        url: Wikipedia URL (e.g., "https://pt.wikipedia.org/wiki/Page_Title")

    Returns:
        Page title (keeping underscores for REST API compatibility)
    """
    try:
        parsed = urlparse(url)
        path_parts = parsed.path.split('/')
        if len(path_parts) >= 3 and path_parts[1] == 'wiki':
            # URL-decode but keep underscores (REST API expects them)
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
    get_html: bool = True,
    rate_limit: float = 0.8
) -> Tuple[Optional[BeautifulSoup], str, List[str], Optional[Dict]]:
    """
    Fetch Wikipedia page content using Wikimedia REST API.

    Args:
        page_title: Wikipedia page title
        lang: Language code
        get_html: If True, fetch HTML content (needed for captions)
        rate_limit: Seconds to wait before making request

    Returns:
        Tuple of (parsed HTML soup or None, final page title, categories list, page data dict)
    """
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Rate limiting
            current_limit = get_current_rate_limit()
            time.sleep(current_limit)

            # URL encode title (safe characters: keep underscores)
            encoded_title = quote(page_title, safe='')
            base_url = f"https://api.wikimedia.org/core/v1/wikipedia/{lang}"

            # Fetch page metadata
            page_url = f"{base_url}/page/{encoded_title}"
            response = API_SESSION.get(page_url, timeout=15)
            response.raise_for_status()
            page_data = response.json()

            final_title = page_data.get('title', page_title)

            # Get HTML if needed
            soup = None
            if get_html:
                time.sleep(current_limit)
                html_url = f"{base_url}/page/{encoded_title}/html"
                html_response = API_SESSION.get(html_url, timeout=15)
                html_response.raise_for_status()
                soup = BeautifulSoup(html_response.text, 'html.parser')

            # Extract categories from HTML
            categories = extract_categories_from_html(soup) if soup else []

            return soup, final_title, categories, page_data

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Too Many Requests - adjust rate limit
                adjust_rate_limit_on_429()
                retry_count += 1
                if retry_count < max_retries:
                    print(f"  [RETRY {retry_count}/{max_retries}] Retrying '{page_title}' with new rate limit...")
                    continue
                else:
                    print(f"  [ERROR] Max retries reached for '{page_title}'")
                    return None, page_title, [], None
            elif e.response.status_code == 404:
                print(f"  [ERROR] Page not found: '{page_title}'")
                return None, page_title, [], None
            else:
                raise

        except Exception as e:
            # Check if it's a 429 hidden in another exception
            if '429' in str(e) or 'Too Many Requests' in str(e):
                adjust_rate_limit_on_429()
                retry_count += 1
                if retry_count < max_retries:
                    print(f"  [RETRY {retry_count}/{max_retries}] Retrying '{page_title}' with new rate limit...")
                    continue

            print(f"  [ERROR] Failed to fetch page '{page_title}': {e}")
            return None, page_title, [], None

    return None, page_title, [], None


def extract_categories_from_html(soup: Optional[BeautifulSoup]) -> List[str]:
    """
    Extract categories from HTML page.

    Args:
        soup: Parsed HTML

    Returns:
        List of category names
    """
    if not soup:
        return []

    categories = []

    # Look for category links in the HTML
    # Wikimedia HTML includes category links with rel="mw:PageProp/Category"
    for link in soup.find_all('link', rel='mw:PageProp/Category'):
        href = link.get('href', '')
        # Extract category name from href like "./Categoria:Something"
        if ':' in href:
            category = href.split(':', 1)[1].replace('_', ' ')
            categories.append(category)

    # Fallback: look for category links in the page
    if not categories:
        for link in soup.find_all('a', href=True):
            href = link['href']
            if '/Categoria:' in href or '/Category:' in href:
                match = re.search(r'/(?:Categoria|Category):(.+)', href)
                if match:
                    category = unquote(match.group(1)).replace('_', ' ')
                    categories.append(category)

    return list(set(categories))  # Remove duplicates


def get_links_from_list_page_rest(
    page_title: str,
    lang: str = 'pt',
    rate_limit: float = 0.8
) -> Set[str]:
    """
    Extract all valid article links from a Wikipedia list page by parsing wikitext.
    Uses /page/{title} endpoint (avoids 403 errors from /html endpoint).
    Only used for list pages, not for entity pages.

    Args:
        page_title: Wikipedia page title
        lang: Language code
        rate_limit: Seconds to wait between requests

    Returns:
        Set of absolute article URLs
    """
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Rate limiting
            current_limit = get_current_rate_limit()
            time.sleep(current_limit)

            # URL encode title
            encoded_title = quote(page_title, safe='')
            base_url = f"https://api.wikimedia.org/core/v1/wikipedia/{lang}"
            page_url = f"{base_url}/page/{encoded_title}"

            # Get page data (includes wikitext source)
            response = API_SESSION.get(page_url, timeout=15)
            response.raise_for_status()
            page_data = response.json()

            # Extract wikitext source
            source = page_data.get('source', '')
            if not source:
                print(f"  [WARNING] No wikitext source found for '{page_title}'")
                return set()

            all_links = set()

            # Extract internal wiki links using regex: [[Page Title]] or [[Page Title|Display Text]]
            # Pattern matches: [[Link]] or [[Link|Text]]
            link_pattern = re.compile(r'\[\[([^\[\]|#]+?)(?:\|[^\[\]]*?)?\]\]')

            for match in link_pattern.finditer(source):
                link_title = match.group(1).strip()

                # Skip empty links
                if not link_title:
                    continue

                # Strip leading colon (used for interwiki/category links: [[:Category:Foo]])
                # This is important to properly detect namespace prefixes
                if link_title.startswith(':'):
                    link_title = link_title[1:]

                # Skip if contains "Lista" or "lista"
                if 'lista' in link_title.lower() or 'list of' in link_title.lower():
                    continue

                # NOTE: Date filtering moved to Phase 2 (analyze_entities) for accurate statistics
                # if is_date_related_page(link_title, lang):
                #     continue

                # Skip interwiki links (other language wikis, wikidata, commons, etc.)
                # Format: :d:Q123, :it:Pagina, :commons:Category:Foo
                interwiki_prefixes = [
                    'd:', 'wikidata:',  # Wikidata
                    'commons:', 'c:',   # Wikimedia Commons
                    # Language codes (common ones)
                    'en:', 'es:', 'fr:', 'de:', 'it:', 'pt:', 'ru:', 'ja:', 'zh:', 'ar:',
                    'nl:', 'pl:', 'sv:', 'fi:', 'no:', 'da:', 'cs:', 'hu:', 'ro:', 'tr:',
                    'uk:', 'el:', 'bg:', 'sr:', 'hr:', 'sk:', 'sl:', 'lt:', 'lv:', 'et:',
                    'ca:', 'gl:', 'eu:', 'is:', 'ga:', 'cy:', 'mt:', 'sq:', 'mk:', 'bs:',
                    'ko:', 'vi:', 'th:', 'id:', 'ms:', 'tl:', 'hi:', 'bn:', 'ta:', 'te:',
                    'he:', 'fa:', 'ur:', 'yi:', 'hy:', 'ka:', 'az:', 'uz:', 'kk:', 'ky:',
                ]

                skip_interwiki = False
                for prefix in interwiki_prefixes:
                    if link_title.lower().startswith(prefix):
                        skip_interwiki = True
                        break

                if skip_interwiki:
                    continue

                # Skip namespace pages (files, special pages, etc.)
                # NOTE: Category filtering moved to Phase 2 (analyze_entities) for accurate statistics
                namespace_patterns = [
                    'ficheiro:', 'file:', 'imagem:', 'image:',  # Files
                    # 'categoria:', 'category:',                   # Categories - MOVED TO PHASE 2
                    'especial:', 'special:',                     # Special pages
                    'wikipedia:', 'wikipédia:',                  # Wikipedia namespace
                    'ajuda:', 'help:',                           # Help pages
                    'portal:',                                   # Portal namespace
                    'predefinição:', 'template:',                # Templates
                    'discussão:', 'talk:',                       # Talk pages
                    'usuário:', 'user:',                         # User pages
                    'módulo:', 'module:',                        # Lua modules
                    'mediawiki:',                                # MediaWiki namespace
                ]

                skip_namespace = False
                for pattern in namespace_patterns:
                    if link_title.lower().startswith(pattern):
                        skip_namespace = True
                        break

                if skip_namespace:
                    continue

                # Convert spaces to underscores for URL
                page_path = link_title.replace(' ', '_')

                # Convert to absolute URL
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


def get_page_media_rest(
    page_title: str,
    lang: str = 'pt',
    rate_limit: float = 0.8
) -> List[Dict]:
    """
    Get all media files from a Wikipedia page using REST API.

    Args:
        page_title: Wikipedia page title
        lang: Language code
        rate_limit: Seconds to wait between requests

    Returns:
        List of image metadata dicts
    """
    try:
        # Rate limiting
        current_limit = get_current_rate_limit()
        time.sleep(current_limit)

        # URL encode title (safe characters: keep underscores)
        encoded_title = quote(page_title, safe='')
        base_url = f"https://api.wikimedia.org/core/v1/wikipedia/{lang}"
        media_url = f"{base_url}/page/{encoded_title}/media"

        response = API_SESSION.get(media_url, timeout=15)
        response.raise_for_status()
        data = response.json()

        images = []
        for item in data.get('items', []):
            if item.get('type') != 'image':
                continue

            title = item.get('title', '')
            # Filter by extension - only accept PNG, JPG, JPEG
            if not title.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            # Get the best available image URL
            srcset = item.get('srcset', [])
            url = None
            if srcset:
                # Get highest resolution
                url = srcset[-1].get('src')
            if not url:
                url = item.get('url')

            if url:
                images.append({
                    'title': title,
                    'url': url,
                    'srcset': srcset,
                    'structured': item.get('structured', {})
                })

        return images

    except Exception as e:
        print(f"  [WARNING] Failed to get media for '{page_title}': {e}")
        return []


def get_file_metadata_local_wiki(
    filename: str,
    lang: str = 'pt',
    rate_limit: float = 0.8
) -> Optional[Dict]:
    """
    Get file metadata from local Wikipedia using MediaWiki Action API.
    Used for files that are hosted locally on language-specific Wikipedias (not on Commons).

    Args:
        filename: Image filename (e.g., "Ficheiro:Example.jpg" or "Example.jpg")
        lang: Language code (e.g., 'pt', 'en', 'es')
        rate_limit: Seconds to wait between requests

    Returns:
        Metadata dict with dimensions, size, mime type, etc.
    """
    try:
        # Rate limiting
        current_limit = get_current_rate_limit()
        time.sleep(current_limit)

        # Clean filename (remove "Ficheiro:" or "File:" prefix)
        if ':' in filename:
            clean_name = filename.split(':', 1)[1]
        else:
            clean_name = filename

        # Build MediaWiki Action API query
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

        # Extract image info from response
        pages = data.get('query', {}).get('pages', {})
        if not pages:
            return None

        # Get first (and only) page
        page = list(pages.values())[0]

        # Check if page exists (negative page ID means not found)
        if int(page.get('pageid', -1)) < 0:
            return None

        # Get imageinfo
        imageinfo = page.get('imageinfo', [])
        if not imageinfo:
            return None

        info = imageinfo[0]

        # Extract file extension from original filename
        file_ext = None
        if '.' in clean_name:
            file_ext = clean_name.rsplit('.', 1)[1].lower()

        return {
            'source_url': info.get('url'),
            'width': info.get('width'),
            'height': info.get('height'),
            'mime': file_ext,
            'size_bytes': info.get('size'),
            'original_url': info.get('url')
        }

    except Exception as e:
        print(f"  [WARNING] Failed to get local file metadata for '{filename}': {e}")
        return None


def get_file_metadata_rest(
    filename: str,
    lang: str = 'pt',
    rate_limit: float = 0.8
) -> Optional[Dict]:
    """
    Get detailed file metadata, trying Commons first, then local Wikipedia.

    This function tries to fetch file metadata in the following order:
    1. Wikimedia Commons (for freely licensed files)
    2. Local language-specific Wikipedia (for non-free/local files)

    Args:
        filename: Image filename (e.g., "Ficheiro:Example.jpg" or "Example.jpg")
        lang: Language code (e.g., 'pt', 'en', 'es')
        rate_limit: Seconds to wait between requests

    Returns:
        Metadata dict with dimensions, size, mime type, etc.
    """
    try:
        # Rate limiting
        current_limit = get_current_rate_limit()
        time.sleep(current_limit)

        # Clean filename (remove "Ficheiro:" or "File:" prefix)
        if ':' in filename:
            clean_name = filename.split(':', 1)[1]
        else:
            clean_name = filename

        # URL encode (safe characters: keep underscores)
        encoded_name = quote(clean_name, safe='')

        # TRY 1: Use Commons API endpoint (not language-specific)
        file_url = f"https://api.wikimedia.org/core/v1/commons/file/File:{encoded_name}"

        response = API_SESSION.get(file_url, timeout=15)
        response.raise_for_status()
        data = response.json()

        # Extract file extension from original filename
        file_ext = None
        if '.' in clean_name:
            file_ext = clean_name.rsplit('.', 1)[1].lower()

        # Get preferred version (usually a reasonable size)
        preferred = data.get('preferred', {})
        original = data.get('original', {})

        return {
            'source_url': preferred.get('url') or original.get('url'),
            'width': preferred.get('width') or original.get('width'),
            'height': preferred.get('height') or original.get('height'),
            'mime': file_ext,  # Store file extension, not mediatype
            'size_bytes': original.get('size'),
            'original_url': original.get('url')
        }

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            # File not found on Commons - try local Wikipedia
            print(f"  [INFO] '{filename}' not on Commons, trying local {lang}.wikipedia.org...")
            return get_file_metadata_local_wiki(filename, lang, rate_limit)
        else:
            print(f"  [WARNING] Failed to get file metadata for '{filename}': {e}")
            return None

    except Exception as e:
        print(f"  [WARNING] Failed to get file metadata for '{filename}': {e}")
        return None


def download_image_from_url(image_url: str) -> Optional[bytes]:
    """
    Download image from a direct URL (e.g., upload.wikimedia.org).
    This is a CDN download - NO rate limiting needed!

    Args:
        image_url: Direct URL to the image file

    Returns:
        Image bytes, or None if failed
    """
    try:
        # NO time.sleep() here - this is just a CDN download, not an API call!
        img_response = API_SESSION.get(image_url, timeout=30)
        img_response.raise_for_status()
        return img_response.content
    except Exception as e:
        print(f"  [WARNING] Failed to download image from '{image_url}': {e}")
        return None


def download_images_parallel(
    image_metadata_list: List[Dict],
    configured_max_workers: int = 10
) -> List[Tuple[Dict, Optional[bytes]]]:
    """
    Download multiple images in parallel from their URLs.
    Since these are CDN downloads (not API calls), we can parallelize without rate limits!

    Args:
        image_metadata_list: List of metadata dicts, each containing 'source_url'
        configured_max_workers: Maximum number of parallel download threads

    Returns:
        List of tuples: (metadata, image_bytes or None)
    """
    results = []
    num_images = len(image_metadata_list)
    if num_images == 0:
        return []

    # Dynamically set max_workers
    actual_max_workers = min(num_images, configured_max_workers)

    with ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
        # Submit all download tasks
        future_to_metadata = {
            executor.submit(download_image_from_url, metadata['source_url']): metadata
            for metadata in image_metadata_list
        }

        # Collect results as they complete
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
# LIST PAGE PROCESSING - Extract Links
# =============================================================================

def is_category_page(url: str) -> bool:
    """
    Check if URL is a category page.

    Args:
        url: Wikipedia URL

    Returns:
        True if URL is a category page, False otherwise
    """
    # Check for category pages (e.g., https://pt.wikipedia.org/wiki/Categoria:Something)
    return '/wiki/Categoria:' in url or '/wiki/Category:' in url


def is_date_related_page(title: str, lang: str = 'pt') -> bool:
    """
    Check if page title is date-related (years, day pages).

    Filters out:
    - Titles that are only numbers (years: "1999", "2023")
    - Titles with day-month patterns (e.g., "12 de Janeiro", "January 12")

    Args:
        title: Page title to check
        lang: Language code for month names

    Returns:
        True if page is date-related, False otherwise
    """
    # Month names for different languages
    months_by_lang = {
        'pt': ['janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho', 'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro'],
        'en': ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'],
        'de': ['januar', 'februar', 'märz', 'april', 'mai', 'juni', 'juli', 'august', 'september', 'oktober', 'november', 'dezember'],
        'es': ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre'],
    }

    months = [m.lower() for m in months_by_lang.get(lang, [])]

    # Check if title is only numbers (years: "1999", "2023", etc.)
    if re.match(r'^\d+$', title.strip()):
        return True

    # Check for "de [month]" pattern (day pages like "12 de Janeiro")
    for month in months:
        if f'de {month}' in title:
            return True
        # Check for "[month] day" pattern (e.g., "January 12")
        if re.search(fr'{month}\s+\d+', title.lower()):
            return True

    return False


def entity_already_processed(entity_title: str, entities_dir: str) -> bool:
    """
    Check if entity has already been processed.

    Args:
        entity_title: Wikipedia page title
        entities_dir: Directory where entities are stored

    Returns:
        True if entity directory with entity_data.json exists, False otherwise
    """
    entity_dir = os.path.join(entities_dir, sanitize_filename(entity_title))
    entity_data_path = os.path.join(entity_dir, "entity_data.json")
    return os.path.exists(entity_data_path)


def remove_entity_directory(entity_title: str, entities_dir: str) -> bool:
    """
    Remove an entity directory and all its contents.
    Used for cleaning up entities that don't pass current filters.

    Args:
        entity_title: Wikipedia page title
        entities_dir: Directory where entities are stored

    Returns:
        True if directory was removed, False if it didn't exist
    """
    import shutil
    entity_dir = os.path.join(entities_dir, sanitize_filename(entity_title))
    if os.path.exists(entity_dir):
        try:
            shutil.rmtree(entity_dir)
            return True
        except Exception as e:
            print(f"  [ERROR] Failed to remove directory for '{entity_title}': {e}")
            return False
    return False


# =============================================================================
# ENTITY PAGE PROCESSING - Extract Content & Images
# =============================================================================

def find_first_infobox_image(soup: BeautifulSoup) -> Optional[str]:
    """
    Find the main infobox image filename.
    Only accepts PNG, JPG, JPEG files.

    Args:
        soup: Parsed HTML of entity page

    Returns:
        Image filename (e.g., "Ficheiro:Example.jpg") or None
    """
    infobox = soup.find('table', class_='infobox')
    if not infobox:
        return None

    # Look for image link in infobox (more reliable than img src)
    img_link = infobox.find('a', class_=['image', 'mw-file-description'], href=True)
    if img_link:
        match = re.search(r'/(?:Ficheiro|File):(.+)', img_link['href'])
        if match:
            filename = f"Ficheiro:{unquote(match.group(1))}"
            # Only accept PNG, JPG, JPEG
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                return filename

    # Fallback: try extracting from img src
    img = infobox.find('img')
    if img and img.has_attr('src'):
        src = img['src']
        # Example: //upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Example.jpg/220px-Example.jpg
        match = re.search(r'/([^/]+\.(?:jpg|jpeg|png))', src, re.IGNORECASE)
        if match:
            raw_filename = unquote(match.group(1))
            clean_filename = re.sub(r'^\d+px-', '', raw_filename)
            return f"Ficheiro:{clean_filename}"

    return None


def page_has_any_images(soup: BeautifulSoup) -> bool:
    """
    Check if page has ANY meaningful images (infobox OR content images).

    Args:
        soup: Parsed HTML of entity page

    Returns:
        True if page has at least one content image, False otherwise
    """
    # First check infobox
    if find_first_infobox_image(soup):
        return True

    # If no infobox image, check for images in main content
    content_div = soup.find('div', class_='mw-parser-output')
    if not content_div:
        return False

    # Look for content images (not icons/logos)
    for img in content_div.find_all('img'):
        if not img.has_attr('src'):
            continue

        src = img['src']

        # Skip icons, logos, and UI elements
        if any(skip in src.lower() for skip in ['icon', 'logo', 'edit', 'magnify', 'crystal']):
            continue

        # Must be from Wikimedia commons/upload
        if 'upload.wikimedia.org' in src:
            return True

    return False


def clean_references(text: str) -> str:
    """
    Remove Wikipedia reference tags (<ref>...</ref>) from text.

    Args:
        text: Text containing potential reference tags

    Returns:
        Text with all reference tags removed
    """
    # Remove <ref>...</ref> tags (including self-closing <ref ... />)
    # Match both <ref>content</ref> and <ref name="..." /> formats
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<ref[^>]*?/>', '', text, flags=re.IGNORECASE)
    return text.strip()


def extract_images_from_wikitext_section(wikitext: str, lang: str = 'pt') -> Tuple[List[Dict], str]:
    """
    Extract image information (filename and caption) from wikitext and remove the image markup from text.
    Also removes template transclusions {{...}}.

    Args:
        wikitext: Wikipedia source text (wikitext format)
        lang: Language code to determine image prefixes.

    Returns:
        Tuple of (list of dicts with 'filename' and 'caption', cleaned text without image markup and templates)
    """
    images = []

    # Collect all image positions and data (no text modification yet)
    image_ranges = []  # List of (start, end) tuples to remove
    lang_config = get_lang_config(lang)
    image_prefixes = lang_config['image_prefixes']

    i = 0
    while i < len(wikitext):
        # Look for [[ followed by an image prefix
        if wikitext[i:i+2] == '[[':
            # Check if this is an image link
            rest = wikitext[i+2:].lower()
            is_image = False
            prefix_len = 0

            for prefix in image_prefixes:
                if rest.startswith(prefix):
                    is_image = True
                    prefix_len = len(prefix)
                    break

            if is_image:
                # Extract the full image markup (handling nested brackets)
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
                    # Found matching ]]
                    end = j
                    full_markup = wikitext[start:end]

                    # Parse the image markup - Remove [[ and ]]
                    inner = full_markup[2:-2]

                    # Split by | carefully handling nested brackets
                    parts = []
                    current_part = []  # Use list for efficient string building
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

                    # First part after prefix is the filename
                    filename = parts[0][prefix_len:].strip() if len(parts) > 0 else ''

                    # Extract caption (typically the last non-option parameter)
                    caption = ''
                    if len(parts) > 1:
                        # Common image options to skip
                        options = ['thumb', 'thumbnail', 'frame', 'miniaturadaimagem', 'miniatura',
                                   'left', 'right', 'center', 'centre', 'none',
                                   'border', 'frameless', 'sem_moldura']

                        for part in reversed(parts[1:]):
                            part_lower = part.strip().lower()
                            # Skip if it's a size (e.g., "200px", "x300px")
                            if 'px' in part_lower:
                                continue
                            # Skip if it's a known option
                            if part_lower in options:
                                continue
                            # Skip if it's a link parameter
                            if part_lower.startswith('link='):
                                continue
                            # Skip if it's alt text parameter
                            if part_lower.startswith('alt='):
                                continue
                            # This is likely the caption
                            caption = part.strip()
                            break

                    # Only accept PNG, JPG, JPEG files
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Clean nested wiki links from caption
                        cleaned_caption = caption
                        if cleaned_caption:
                            # Remove [[ and ]] but keep content
                            wiki_link_pattern = re.compile(r'\[\[([^\]|]+?)(?:\|([^\]]+?))?\]\]')
                            cleaned_caption = wiki_link_pattern.sub(lambda m: m.group(2) if m.group(2) else m.group(1), cleaned_caption)
                            # Remove <ref> tags from caption
                            cleaned_caption = clean_references(cleaned_caption)

                        images.append({
                            'filename': filename,
                            'caption': cleaned_caption
                        })

                    # Record this range for removal
                    image_ranges.append((start, end))

                    # Skip past this image
                    i = end
                    continue

        i += 1

    # Build cleaned text in one pass by copying segments between images
    if image_ranges:
        cleaned_parts = []
        last_end = 0

        for start, end in image_ranges:
            # Add text before this image
            cleaned_parts.append(wikitext[last_end:start])
            last_end = end

        # Add remaining text after last image
        cleaned_parts.append(wikitext[last_end:])
        cleaned_text = ''.join(cleaned_parts)
    else:
        cleaned_text = wikitext

    # Remove template transclusions: {{...}}
    # Use a simple approach that handles nested braces by removing outermost templates
    while '{{' in cleaned_text:
        # Find the first {{
        start = cleaned_text.find('{{')
        if start == -1:
            break

        # Find the matching }}
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
            # Remove this template
            cleaned_text = cleaned_text[:start] + cleaned_text[end:]
        else:
            # Malformed template, just remove the opening {{
            cleaned_text = cleaned_text[:start] + cleaned_text[start+2:]

    return images, cleaned_text.strip()


def extract_images_from_wikitext(wikitext: str, lang: str = 'pt') -> List[str]:
    """
    Extract image filenames from wikitext by finding [[Ficheiro:...]] patterns.

    Args:
        wikitext: Wikipedia source text (wikitext format)
        lang: Language code to determine image prefixes.

    Returns:
        List of image filenames (e.g., ["Cristiano_Ronaldo.jpg", "Example.png"])
    """
    images = []
    lang_config = get_lang_config(lang)
    prefixes = '|'.join([p.rstrip(':') for p in lang_config['image_prefixes']])

    # Pattern to match: [[Ficheiro:filename.ext|...]] or [[File:filename.ext|...]]
    file_pattern = re.compile(fr'\[\[(?:{prefixes}):([^\]|]+?)(?:\|[^\]]*?)?\]\]', re.IGNORECASE)

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
    Faster but does not include image captions.

    Args:
        page_data: Page data dict from REST API
        lang: Language code for ignored sections.

    Returns:
        List of dicts with {heading, content, images}
    """
    sections_data = []
    lang_config = get_lang_config(lang)
    # Sections to ignore
    IGNORED_SECTIONS = lang_config['ignored_sections']

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
            # Extract images and clean content
            lead_images_info, lead_content_cleaned = extract_images_from_wikitext_section(lead_content, lang)
            # Clean references from lead content
            lead_content_cleaned = clean_references(lead_content_cleaned)
            if lead_content_cleaned:
                sections_data.append({
                    'heading': 'Lead Section',
                    'content': [lead_content_cleaned],
                    'images': lead_images_info  # Store image info (filename + caption) found in this section
                })

        # Extract each section
        for i, match in enumerate(matches):
            section_title = match.group(2).strip()
            # Clean references from section title
            section_title = clean_references(section_title)

            # Skip ignored sections
            if section_title in IGNORED_SECTIONS:
                continue

            # Get content between this heading and the next (or end of text)
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(source)
            section_content = source[start_pos:end_pos].strip()

            # Extract images and clean content (remove image markup and templates)
            section_images_info, section_content_cleaned = extract_images_from_wikitext_section(section_content, lang)
            # Clean references from section content
            section_content_cleaned = clean_references(section_content_cleaned)

            if section_content_cleaned or section_images_info:
                sections_data.append({
                    'heading': section_title,
                    'content': [section_content_cleaned] if section_content_cleaned else [],
                    'images': section_images_info  # Store image info (filename + caption) found in this section
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


def extract_sections_with_images(soup: BeautifulSoup) -> List[Dict]:
    """
    Extract content organized by sections, with associated images.
    Only extracts PNG, JPG, JPEG images.

    Args:
        soup: Parsed HTML of entity page

    Returns:
        List of dicts with {heading, content, images}
    """
    sections = []
    content_div = soup.find('div', class_='mw-parser-output')
    if not content_div:
        return []

    # Sections to ignore
    IGNORED_SECTIONS = {'Bibliografia', 'Referências', 'Ligações externas', 'Ver também', 'Notas'}

    def extract_images_from_element(element):
        """Extract all image links from an element. Only PNG, JPG, JPEG."""
        images = []
        # Find all image links in this element
        for link in element.find_all('a', class_=['image', 'mw-file-description'], href=True):
            match = re.search(r'/(?:Ficheiro|File):(.+)', link['href'])
            if match:
                filename = f"Ficheiro:{unquote(match.group(1))}"
                # Only accept PNG, JPG, JPEG
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Find caption if available
                    parent = link.find_parent(['div', 'figure'])
                    caption = ""
                    if parent:
                        caption_tag = parent.select_one('figcaption, .thumbcaption')
                        if caption_tag:
                            caption = caption_tag.get_text(strip=True)
                    images.append({'filename': filename, 'caption': caption})
        return images

    # First, get the lead section (before first h2)
    lead_section = {'heading': 'Lead Section', 'content': [], 'images': []}

    for element in content_div.find_all(recursive=False):
        if element.name == 'h2':
            break  # End of lead section
        if element.name == 'p':
            text = element.get_text(separator=' ', strip=True)
            if text:
                lead_section['content'].append(text)
        elif element.name in ['ul', 'ol']:
            list_items = []
            for li in element.find_all('li', recursive=False):
                item_text = li.get_text(separator=' ', strip=True)
                if item_text:
                    list_items.append(f"• {item_text}")
            if list_items:
                lead_section['content'].append('\n'.join(list_items))

        # Extract images from lead section
        lead_section['images'].extend(extract_images_from_element(element))

    if lead_section['content'] or lead_section['images']:
        sections.append(lead_section)

    # Now process the rest of the page section by section
    current_section = None
    skip_current_section = False

    for element in content_div.find_all(['h2', 'h3', 'h4', 'p', 'ul', 'ol', 'div', 'figure', 'table']):
        if element.name in ['h2', 'h3', 'h4']:
            if current_section and not skip_current_section and (current_section['content'] or current_section['images']):
                sections.append(current_section)

            heading_text = element.get_text(separator=' ', strip=True)
            heading_text = re.sub(r'\[editar\]', '', heading_text).strip()
            skip_current_section = heading_text in IGNORED_SECTIONS
            current_section = {'heading': heading_text, 'content': [], 'images': []}

        elif current_section is not None and not skip_current_section:
            if element.name == 'p':
                text = element.get_text(separator=' ', strip=True)
                if text:
                    current_section['content'].append(text)
            elif element.name in ['ul', 'ol']:
                list_items = []
                for li in element.find_all('li', recursive=False):
                    item_text = li.get_text(separator=' ', strip=True)
                    if item_text:
                        list_items.append(f"• {item_text}")
                if list_items:
                    current_section['content'].append('\n'.join(list_items))

            # Extract all images from any element (div, figure, table, etc.)
            current_section['images'].extend(extract_images_from_element(element))

    if current_section and not skip_current_section and (current_section['content'] or current_section['images']):
        sections.append(current_section)

    return sections


def _get_extension_from_url(url: str) -> str:
    """Extract file extension from URL."""
    parsed = urlparse(url)
    path = parsed.path
    ext = os.path.splitext(path)[1].lower()
    # Default to .jpg if no extension
    return ext if ext in ['.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp'] else '.jpg'


def extract_infobox_image_from_wikitext(wikitext: str, lang: str = 'pt') -> Optional[Dict[str, str]]:
    """
    Extracts the main image and caption from the first infobox in the wikitext.

    An infobox typically starts with "{{Info" and contains parameters for the
    main image and its caption.

    Args:
        wikitext: The wikitext source of the page.
        lang: Language code for keyword matching.

    Returns:
        A dictionary with 'filename' and 'caption', or None if not found.
    """
    lang_config = get_lang_config(lang)
    image_params = lang_config['infobox_image_param']
    caption_params = lang_config['infobox_caption_param']

    # Find the start of the first infobox template (e.g., "{{Info/Futebol", "{{Info/Biografia")
    infobox_match = re.search(r'\{\{(?:info|infobox)', wikitext, re.IGNORECASE)
    if not infobox_match:
        return None

    # Find the content of this template, handling nested braces
    start = infobox_match.start()
    depth = 0
    end = -1
    for i in range(start, len(wikitext) - 1):
        if wikitext[i:i+2] == '{{':
            depth += 1
        elif wikitext[i:i+2] == '}}':
            depth -= 1
            if depth == 0:
                end = i + 2
                break

    if end == -1:
        return None

    infobox_content = wikitext[start:end]

    # Extract image filename from "imagem = ..."
    # Try all possible image parameter names
    filename = None
    if isinstance(image_params, list):
        for param in image_params:
            image_match = re.search(fr'\|\s*{param}\s*=\s*([^\s][^|\n]*?)\s*(?=[|\n])', infobox_content, re.IGNORECASE)
            if image_match:
                filename = image_match.group(1).strip()
                break
    else: # Fallback for single string config
        image_match = re.search(fr'\|\s*{image_params}\s*=\s*([^\s][^|\n]*?)\s*(?=[|\n])', infobox_content, re.IGNORECASE)
        if image_match:
            filename = image_match.group(1).strip()

    if not filename:
        return None

    # Extract caption from "imagem_legenda = ..."
    caption = ''
    caption_params_list = caption_params if isinstance(caption_params, list) else [caption_params]
    for param in caption_params_list:
        caption_match = re.search(fr'\|\s*{param}\s*=\s*([^|\n]+)', infobox_content)
        if caption_match:
            caption = caption_match.group(1).strip()
            break

    # Clean caption (remove wikitext links, etc.)
    cleaned_caption = re.sub(r'\[\[(?:[^|\]]+\|)?([^\]]+)\]\]', r'\1', caption)
    cleaned_caption = re.sub(r'\{\{[^}]+\}\}', '', cleaned_caption).strip()

    return {'filename': filename, 'caption': cleaned_caption}

def _fetch_images_for_language(
    entity_title: str,
    lang: str,
    rate_limit: float
) -> Tuple[List[Dict], List[Dict]]:
    """
    Fetches all image information for a given entity in a specific language.

    Args:
        entity_title: The title of the entity.
        lang: The language to fetch images from.
        rate_limit: The rate limit for API calls.

    Returns:
        A tuple containing:
        - A list of image metadata dictionaries to be processed.
        - A list of section dictionaries for this language.
    """
    _, _, _, page_data = get_page_content_rest(
        entity_title,
        lang=lang,
        get_html=False,
        rate_limit=rate_limit
    )

    if not page_data or not page_data.get('source'):
        return [], []

    wikitext = page_data['source']

    # Extract infobox image and other images from wikitext
    infobox_image_info = extract_infobox_image_from_wikitext(wikitext, lang)
    all_image_filenames = extract_images_from_wikitext(wikitext, lang)
    sections = extract_content_from_page_source(page_data, lang)

    # Build mapping from filename to section and caption
    image_to_section = {}
    image_to_caption = {}
    for section in sections:
        for img_info in section.get('images', []):
            image_to_section[img_info['filename']] = section['heading']
            image_to_caption[img_info['filename']] = img_info.get('caption', '')

    images_to_fetch = []
    processed_filenames = set()

    # Prioritize infobox image
    if infobox_image_info and infobox_image_info['filename']:
        filename = infobox_image_info['filename']
        if filename not in processed_filenames:
            images_to_fetch.append({
                'filename': filename, 'is_primary': True, 'role': 'infobox_image',
                'section': 'Lead Section', 'caption': infobox_image_info.get('caption', ''),
                'lang': lang
            })
            processed_filenames.add(filename)

    # Add other images
    for filename in all_image_filenames:
        if filename not in processed_filenames:
            images_to_fetch.append({
                'filename': filename, 'is_primary': (len(images_to_fetch) == 0), 'role': 'wikitext_image',
                'section': image_to_section.get(filename), 'caption': image_to_caption.get(filename, ''),
                'lang': lang
            })
            processed_filenames.add(filename)

    return images_to_fetch, sections

def process_entity_page(
    entity_title: str,
    output_dir: str,
    entity_url: str,
    origin_lists: List[str],
    languages: List[str],
    rate_limit: float = 0.8,
    extract_captions: bool = True,
    parallel_downloads: bool = True,
    download_workers: int = 10,
    quiet: bool = False
) -> Optional[str]:
    """
    Process a single entity page: download images and extract content.
    Only saves entities that have images.

    Args:
        entity_title: Wikipedia page title
        output_dir: Directory to save entity data
        entity_url: Original URL of the entity
        origin_lists: List of list URLs that referenced this entity
        languages: List of language codes to process (first is primary).
        rate_limit: Seconds to wait between requests
        extract_captions: If True, parse HTML for image captions (slower)
        parallel_downloads: If True, download images in parallel (faster)
        download_workers: Number of parallel workers for image downloads
        quiet: If True, suppress progress messages

    Returns:
        None if successful with images, 'no_images' if no images found, entity_title if failed
    """
    if not quiet:
        print(f"  Processing: {entity_title}")

    primary_lang = languages[0]

    # Fetch primary language content for text and categories
    soup, final_title, categories, page_data = get_page_content_rest(
        entity_title,
        lang=primary_lang,
        get_html=False,  # Always False for entity pages
        rate_limit=rate_limit
    )

    if page_data is None:
        if not quiet:
            print("    [SKIP] Could not retrieve page")
        return entity_title

    # Check if this was a redirect and notify
    if final_title != entity_title:
        if not quiet:
            print(f"    [REDIRECT] {entity_title} → {final_title}")
        # Use the final title for all subsequent operations
        entity_title = final_title

        # Check if the redirected entity is already processed
        entities_base_dir = os.path.join(output_dir, "entities")
        if entity_already_processed(entity_title, entities_base_dir):
            if not quiet:
                print(f"    [SKIP] Already processed as '{entity_title}'")
            return None  # Success - already exists

    # Extract text sections from the primary language page
    sections = extract_content_from_page_source(page_data, primary_lang) if page_data else []

    # --- Multi-language image fetching ---
    all_images_to_fetch = []
    all_image_sections = {} # Maps filename to section for primary lang

    for lang in languages:
        if not quiet:
            print(f"    Fetching images for language: [{lang}]")

        lang_images, lang_sections = _fetch_images_for_language(final_title, lang, rate_limit)
        all_images_to_fetch.extend(lang_images)

        # Store section info for primary language images
        if lang == primary_lang:
            for section_data in lang_sections:
                for img_info in section_data.get('images', []):
                    all_image_sections[img_info['filename']] = section_data['heading']

    if not all_images_to_fetch:
        if not quiet:
            print("    [NO IMAGES] No images found across all languages. Skipping.")
        return 'no_images'

    if not quiet:
        print(f"    Found {len(all_images_to_fetch)} total images across {len(languages)} language(s)")

    # Fetch metadata for all images from Commons API
    metadata_list = []
    for img_info in all_images_to_fetch:
        # Get metadata from Commons REST API
        metadata = get_file_metadata_rest(img_info['filename'], img_info['lang'], rate_limit)
        if metadata:
            metadata['image_role'] = img_info['role']
            metadata['is_primary'] = img_info['is_primary']
            metadata['section'] = img_info['section']
            metadata['caption'] = img_info['caption']
            metadata['original_filename'] = img_info['filename']
            metadata['lang'] = img_info['lang']
            metadata_list.append(metadata)

    # Download images (parallel or sequential)
    if parallel_downloads:
        if not quiet and len(metadata_list) > 1:
            print(f"    Downloading {len(metadata_list)} images in parallel...")
        download_results = download_images_parallel(metadata_list, configured_max_workers=download_workers)
    else:
        if not quiet and len(metadata_list) > 1:
            print(f"    Downloading {len(metadata_list)} images sequentially...")
        download_results = []
        for metadata in metadata_list:
            image_bytes = download_image_from_url(metadata['source_url'])
            download_results.append((metadata, image_bytes))

    # Save all downloaded images
    images_data = []
    for metadata, image_bytes in download_results:
        # Create entity and language-specific directories
        entity_dir = os.path.join(output_dir, sanitize_filename(entity_title))
        lang_dir = os.path.join(entity_dir, metadata['lang'])
        os.makedirs(lang_dir, exist_ok=True)
        if not image_bytes:
            continue

        # Determine filename
        if 'original_filename' in metadata and ':' in metadata['original_filename']:
            fname = metadata['original_filename'].split(":", 1)[1]
        else:
            fname = metadata.get('original_filename', 'image')

        base_filename = sanitize_filename(os.path.splitext(fname)[0])
        ext = _get_extension_from_url(metadata['source_url'])

        image_path = os.path.join(lang_dir, f"{base_filename}{ext}")
        meta_path = os.path.join(lang_dir, f"{base_filename}.json")

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
            'image_role': metadata['image_role'],
            'is_primary': metadata['is_primary'],
            'lang': metadata['lang']
        }
        if metadata.get('section'):
            metadata_to_save['section'] = metadata['section']
        if metadata.get('caption'):
            metadata_to_save['caption'] = metadata['caption']

        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_to_save, f, ensure_ascii=False, indent=2)

        images_data.append({
            'original_filename': metadata.get('original_filename', fname),
            'local_filename': os.path.basename(image_path),
            'is_primary': metadata['is_primary'],
            'lang': metadata['lang']
        })

        if not quiet and metadata['is_primary']:
            print(f"    ✓ Main image: {os.path.basename(image_path)}")

    # Check if we actually downloaded any images
    if len(images_data) == 0:
        # No images were downloaded - clean up directory and return no_images
        entity_dir = os.path.join(output_dir, sanitize_filename(entity_title))
        import shutil
        if os.path.exists(entity_dir):
            shutil.rmtree(entity_dir)
        if not quiet:
            print("    [NO IMAGES] Zero images downloaded after filtering")
        return 'no_images'

    # Build mapping from section name to count of downloaded images
    section_image_counts = {}
    for img_data in images_data:
        # Get the original filename to look up its section
        orig_filename = img_data.get('original_filename', '')
        # Remove "Ficheiro:" or "File:" prefix if present
        if ':' in orig_filename:
            clean_filename = orig_filename.split(':', 1)[1]
        else:
            clean_filename = orig_filename

        # Look up which section this image belongs to
        section_name = all_image_sections.get(clean_filename, None)
        if section_name:
            section_image_counts[section_name] = section_image_counts.get(section_name, 0) + 1

    # Save entity data
    entity_data = {
        'title': final_title,
        'url': entity_url,
        'origin_lists': origin_lists,
        'has_images': True,
        'language': primary_lang,
        'categories': categories,
        'sections': [
            {
                'heading': s['heading'],
                'content': s['content'],
                'image_count': section_image_counts.get(s['heading'], 0)  # Count of downloaded images in this section
            } for s in sections
        ],
        'images': images_data,
        'total_images': len(images_data)
    }

    entity_dir = os.path.join(output_dir, sanitize_filename(entity_title))
    entity_data_path = os.path.join(entity_dir, "entity_data.json")
    with open(entity_data_path, 'w', encoding='utf-8') as f:
        json.dump(entity_data, f, ensure_ascii=False, indent=2)

    if not quiet:
        print(f"    ✓ Saved {len(images_data)} images")
    return None  # Success


# =============================================================================
# PARALLEL PROCESSING
# =============================================================================

def process_entity_batch(args):
    """
    Worker function for multiprocessing.
    Processes a batch of entities and writes results to worker-specific files.

    Args:
        args: Tuple of (worker_id, entities_batch, entities_dir, output_dir, lang, rate_limit, extract_captions, api_token)

    Returns:
        Dictionary with results: {processed, no_images, failed}
    """
    worker_id, entities_batch, entities_dir, output_dir, languages, rate_limit, extract_captions, api_token, parallel_downloads, download_workers = args

    # Initialize session for this worker
    configure_rest_api_session(api_token, languages[0])
    configure_rate_limit(bool(api_token), rate_limit)

    # Worker-specific output file for no-images URLs
    no_images_file = os.path.join(output_dir, f"no_images_worker_{worker_id}.txt")

    processed_with_images = []
    skipped_no_images = []
    failed = []

    with open(no_images_file, 'w', encoding='utf-8') as no_img_file:
        # Create progress bar for this worker
        for entity in tqdm(entities_batch,
                          desc=f"Worker {worker_id}",
                          unit="entity",
                          position=worker_id,
                          leave=True):
            result = process_entity_page(
                entity['title'],
                entities_dir,
                entity['url'],
                [],  # Origin lists
                languages,
                rate_limit,
                extract_captions,
                parallel_downloads,
                download_workers,
                quiet=True
            )

            if result is None:
                processed_with_images.append(entity)
            elif result == 'no_images':
                no_img_file.write(f"{entity['url']}\n")
                no_img_file.flush()
                skipped_no_images.append(entity)
            else:
                failed.append(entity)

    return {
        'worker_id': worker_id,
        'processed': processed_with_images,
        'no_images': skipped_no_images,
        'failed': failed
    }


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def fetch_single_list(args):
    """
    Worker function to fetch a single list page.

    Args:
        args: Tuple of (list_url, lang, rate_limit)

    Returns:
        Tuple of (list_url, article_links set)
    """
    list_url, lang, rate_limit = args

    # Extract title from URL
    page_title = extract_title_from_url(list_url)
    if not page_title:
        return (list_url, set())

    # Get links using REST API
    article_links = get_links_from_list_page_rest(page_title, lang=lang, rate_limit=rate_limit)
    return (list_url, article_links)


def collect_all_links_from_lists(
    list_urls: List[str],
    lang: str = 'pt',
    rate_limit: float = 0.8,
    max_workers: int = 5
) -> Dict[str, Set[str]]:
    """
    Collect all unique article links from all list pages using REST API.
    Processes multiple lists in parallel for faster collection.

    Args:
        list_urls: List of Wikipedia list URLs
        lang: Language code
        rate_limit: Seconds to wait between requests (per worker)
        max_workers: Number of parallel workers for fetching lists

    Returns:
        Dictionary mapping list URLs to sets of article URLs
    """
    print(f"\n{'='*80}")
    print("PHASE 1: Collecting all links from lists")
    print(f"{'='*80}\n")
    print(f"Processing {len(list_urls)} lists with {max_workers} parallel workers\n")

    all_links = {}

    # Prepare arguments for parallel processing
    fetch_args = [(url, lang, rate_limit) for url in list_urls]

    # Process lists in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_url = {executor.submit(fetch_single_list, args): args[0] for args in fetch_args}

        # Process results as they complete
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
    output_dir: str
) -> Dict:
    """
    Analyze all collected links to determine what will be processed/skipped.
    Enhanced statistics tracking at each filtering step.

    Args:
        all_links: Dictionary mapping list URLs to article URL sets
        entities_dir: Directory where entities are stored
        output_dir: Directory to save filtered URLs

    Returns:
        Analysis statistics dictionary
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(entities_dir, exist_ok=True)

    # Collect all unique URLs
    all_unique_urls = set()
    for links in all_links.values():
        all_unique_urls.update(links)

    print(f"\n{'='*80}")
    print("PHASE 2: Analyzing entities with enhanced statistics")
    print(f"{'='*80}\n")

    # Statistics tracking
    total_collected = len(all_unique_urls)
    category_pages = []
    date_filtered = []
    already_processed = []
    to_process = []
    cleaned_up = []  # Track entities removed due to failing current filters

    print(f"📊 Total unique entities collected: {total_collected}")

    for url in sorted(all_unique_urls):
        # Check if category page (filter FIRST before extracting title)
        if is_category_page(url):
            # Extract title for logging
            parsed_url = urlparse(url)
            path_parts = parsed_url.path.split('/')
            page_title = unquote(path_parts[2]) if len(path_parts) >= 3 else url

            # Check if already processed and remove if so (cleanup from previous buggy runs)
            if entity_already_processed(page_title, entities_dir):
                if remove_entity_directory(page_title, entities_dir):
                    print(f"  [CLEANUP] Removed category page: {page_title}")
                    cleaned_up.append({'url': url, 'title': page_title, 'reason': 'category'})

            category_pages.append({'url': url, 'title': page_title})
            continue

        # Extract page title from URL
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.split('/')
        if len(path_parts) < 3 or path_parts[1] != 'wiki':
            continue

        page_title = unquote(path_parts[2])

        # Check if date-related
        if is_date_related_page(page_title, 'pt'): # Assuming primary lang for filtering
            # Check if already processed and remove if so (cleanup from previous buggy runs)
            if entity_already_processed(page_title, entities_dir):
                if remove_entity_directory(page_title, entities_dir):
                    print(f"  [CLEANUP] Removed date page: {page_title}")
                    cleaned_up.append({'url': url, 'title': page_title, 'reason': 'date'})

            date_filtered.append({'url': url, 'title': page_title})
        # Check if already processed
        elif entity_already_processed(page_title, entities_dir):
            already_processed.append({'url': url, 'title': page_title})
        else:
            to_process.append({'url': url, 'title': page_title})

    # Print comprehensive statistics
    after_category_filter = total_collected - len(category_pages)
    after_date_filter = after_category_filter - len(date_filtered)
    after_already_processed_filter = after_date_filter - len(already_processed)

    print(f"📊 After category filtering: {after_category_filter} entities (removed {len(category_pages)} category pages)")
    print(f"📊 After date filtering: {after_date_filter} entities (removed {len(date_filtered)} date pages)")
    print(f"📊 After already-processed filtering: {after_already_processed_filter} entities (skipped {len(already_processed)} existing)")

    if cleaned_up:
        print(f"🧹 Cleaned up {len(cleaned_up)} previously-processed entities that failed current filters:")
        category_cleanups = [item for item in cleaned_up if item['reason'] == 'category']
        date_cleanups = [item for item in cleaned_up if item['reason'] == 'date']
        if category_cleanups:
            print(f"   - {len(category_cleanups)} category pages")
        if date_cleanups:
            print(f"   - {len(date_cleanups)} date-related pages")

    print(f"📊 To process: {len(to_process)} entities")
    print()

    # Save category pages
    if category_pages:
        categories_file = os.path.join(output_dir, "categories.txt")
        with open(categories_file, 'w', encoding='utf-8') as f:
            for item in category_pages:
                f.write(f"{item['url']}\n")
        print(f"✓ Category pages saved to: {categories_file}")

    # Save date-filtered URLs
    date_filtered_urls_file = os.path.join(output_dir, "date_filtered_urls.txt")
    with open(date_filtered_urls_file, 'w', encoding='utf-8') as f:
        for item in date_filtered:
            f.write(f"{item['url']}\n")
    print(f"✓ Date-filtered URLs saved to: {date_filtered_urls_file}")

    if date_filtered:
        print("\nDate-related pages (first 10 examples):")
        for item in date_filtered[:10]:
            print(f"  • {item['title']}")
        if len(date_filtered) > 10:
            print(f"  ... and {len(date_filtered) - 10} more")
    print()

    return {
        'total_collected': total_collected,
        'after_category_filter': after_category_filter,
        'after_date_filter': after_date_filter,
        'after_already_processed_filter': after_already_processed_filter,
        'category_pages': category_pages,
        'date_filtered': date_filtered,
        'already_processed': already_processed,
        'to_process': to_process
    }


def main():
    """Main entry point."""
    # Load configuration
    config = load_config()

    output_dir = config.get('output_dir', str(DATA / '01_extraction' / 'newdata' / 'temporal_relevant'))
    languages = config.get('languages', ['pt'])
    lists_file = config.get('lists_file', 'lists.txt')
    rate_limit = config.get('rate_limit_seconds', 0.8)
    num_workers = config.get('num_workers', 4)
    list_fetch_workers = config.get('list_fetch_workers', 5)
    extract_captions = config.get('extract_image_captions', True)
    parallel_downloads = config.get('parallel_image_downloads', True)
    download_workers = config.get('image_download_workers', 10)
    api_token = config.get('wikimedia_api_token')

    if not languages:
        raise ValueError("Configuration error: 'languages' list cannot be empty.")
    primary_lang = languages[0]

    # Initialize REST API session
    configure_rest_api_session(api_token, primary_lang)
    configure_rate_limit(bool(api_token), rate_limit)

    print("="*80)
    print("Wikipedia List Processor - REST API Version")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Languages: {languages} (Primary: {primary_lang})")
    print(f"Lists file: {lists_file}")
    print(f"Rate limit: {rate_limit}s")
    print(f"Entity workers: {num_workers}")
    print(f"List fetch workers: {list_fetch_workers}")
    print(f"Extract captions: {extract_captions}")
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

    # Determine entities directory
    entities_dir = os.path.join(output_dir, "entities")
    os.makedirs(entities_dir, exist_ok=True)

    # PHASE 1: Collect all links from all lists
    all_links = collect_all_links_from_lists(list_urls, primary_lang, rate_limit, list_fetch_workers)

    # PHASE 2: Analyze what needs to be processed
    analysis = analyze_entities(all_links, entities_dir, output_dir)

    # PHASE 3: Process entities
    print(f"\n{'='*80}")
    print("PHASE 3: Processing entities")
    print(f"{'='*80}\n")

    processed_with_images = []
    all_skipped_no_images = []
    failed_entities = []

    if analysis['to_process']:
        print(f"Processing {len(analysis['to_process'])} entities...\n")
        print(f"Using {num_workers} parallel workers\n")

        # Split entities into batches for each worker
        entities_list = analysis['to_process']
        batch_size = len(entities_list) // num_workers + (1 if len(entities_list) % num_workers else 0)
        batches = []

        for i in range(num_workers):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(entities_list))
            if start_idx < len(entities_list):
                batch = entities_list[start_idx:end_idx]
                batches.append((i, batch, entities_dir, output_dir, languages, rate_limit, extract_captions, api_token, parallel_downloads, download_workers))

        print(f"Split {len(entities_list)} entities into {len(batches)} batches\n")

        # Process batches in parallel
        with Pool(processes=num_workers) as pool:
            results = list(pool.map(process_entity_batch, batches))

        # Aggregate results from all workers
        for result in results:
            processed_with_images.extend(result['processed'])
            all_skipped_no_images.extend(result['no_images'])
            failed_entities.extend(result['failed'])

        print("\n✓ All workers completed")
        print(f"  Worker output files: no_images_worker_0.txt to no_images_worker_{len(batches)-1}.txt\n")

    # Final statistics with enhanced tracking
    print(f"\n{'='*80}")
    print("FINAL STATISTICS")
    print(f"{'='*80}")
    print(f"📊 Total unique entities collected: {analysis['total_collected']}")
    print(f"📊 After category filtering: {analysis['after_category_filter']} (-{len(analysis['category_pages'])} category pages)")
    print(f"📊 After date filtering: {analysis['after_date_filter']} (-{len(analysis['date_filtered'])} date pages)")
    print(f"📊 After already-processed filtering: {analysis['after_already_processed_filter']} (-{len(analysis['already_processed'])} existing)")
    print(f"📊 Newly processed: {len(analysis['to_process'])}")
    print(f"   ✓ With images: {len(processed_with_images)}")
    print(f"   ✗ Without images: {len(all_skipped_no_images)}")
    print(f"   ✗ Failed: {len(failed_entities)}")
    print(f"\n📊 Final entities with images: {len(processed_with_images)}")
    print(f"{'='*80}")

    # Save comprehensive statistics
    stats_data = {
        'total_collected': analysis['total_collected'],
        'after_category_filter': analysis['after_category_filter'],
        'after_date_filter': analysis['after_date_filter'],
        'after_already_processed_filter': analysis['after_already_processed_filter'],
        'category_pages_count': len(analysis['category_pages']),
        'date_filtered_count': len(analysis['date_filtered']),
        'already_processed_count': len(analysis['already_processed']),
        'newly_processed': len(analysis['to_process']),
        'with_images': len(processed_with_images),
        'without_images': len(all_skipped_no_images),
        'failed': len(failed_entities),
        'category_pages_list': [item['title'] for item in analysis['category_pages']],
        'date_filtered_list': [item['title'] for item in analysis['date_filtered']],
        'no_images_list': [item['title'] for item in all_skipped_no_images],
        'failed_list': [item['title'] for item in failed_entities]
    }

    stats_file = os.path.join(output_dir, "processing_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Statistics saved to: {stats_file}")

    # Save no-images URLs
    no_images_file = os.path.join(output_dir, "no_images.txt")
    with open(no_images_file, 'w', encoding='utf-8') as f:
        for item in all_skipped_no_images:
            f.write(f"{item['url']}\n")
    print(f"✓ No-images URLs saved to: {no_images_file}")

    # Show rate limit adjustments if any occurred
    if RATE_LIMIT_ADJUSTMENTS > 0:
        print(f"\n⚠️  Rate limit was adjusted {RATE_LIMIT_ADJUSTMENTS} time(s) due to 429 errors")
        print(f"   Final rate limit: {CURRENT_RATE_LIMIT:.3f}s per request")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
