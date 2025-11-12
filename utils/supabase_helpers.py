import os
import urllib.parse
from typing import Mapping, Optional


POOLER_SUFFIX = ".pooler.supabase.com"
SUPABASE_DOMAINS = (".supabase.co", ".supabase.in")


def _extract_project_ref_from_url(url: str) -> Optional[str]:
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return None

    host = parsed.hostname or ""
    host = host.strip().lower()

    for domain in SUPABASE_DOMAINS:
        if host.endswith(domain):
            project_ref = host[: -len(domain)]
            # Supabase project refs are alphanumeric strings without dots
            if project_ref:
                return project_ref

    return None


def get_supabase_project_ref(environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    env = environ or os.environ

    direct_ref = (
        env.get("SUPABASE_PROJECT_REF")
        or env.get("SUPABASE_PROJECT_ID")
        or env.get("SUPABASE_REFERENCE_ID")
    )
    if direct_ref:
        return direct_ref.strip()

    for key in ("SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_URL"):
        url = env.get(key)
        if url:
            project_ref = _extract_project_ref_from_url(url)
            if project_ref:
                return project_ref

    return None


def normalize_pooler_host(host: Optional[str], environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    if not host:
        return host

    normalized_host = host.strip()
    if not normalized_host:
        return None

    if normalized_host.endswith(POOLER_SUFFIX):
        return normalized_host

    project_ref = get_supabase_project_ref(environ)
    if not project_ref or project_ref in normalized_host:
        return normalized_host

    return f"{normalized_host}-{project_ref}"


def normalize_db_url(db_url: str, environ: Optional[Mapping[str, str]] = None) -> str:
    if not db_url:
        return db_url

    try:
        parsed = urllib.parse.urlparse(db_url)
    except ValueError:
        return db_url

    netloc = parsed.netloc
    if not netloc:
        return db_url

    creds = ""
    host_port = netloc
    if "@" in netloc:
        creds, host_port = netloc.split("@", 1)

    host = host_port
    port = None
    if ":" in host_port:
        host, port = host_port.split(":", 1)

    normalized_host = normalize_pooler_host(host, environ)
    if normalized_host == host:
        return db_url

    new_host_port = f"{normalized_host}:{port}" if port else normalized_host
    new_netloc = f"{creds}@{new_host_port}" if creds else new_host_port

    return urllib.parse.urlunparse(parsed._replace(netloc=new_netloc))


def build_supabase_db_url(environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    env = environ or os.environ

    candidate_keys = (
        "SUPABASE_DB_URL",
        "SUPABASE_POSTGRES_URL",
        "SUPABASE_DB_CONNECTION",
        "SUPABASE_CONNECTION_STRING",
        "DB_URL",
        "DATABASE_URL",
    )

    for key in candidate_keys:
        value = env.get(key)
        if value:
            return normalize_db_url(value, env)

    host = normalize_pooler_host(env.get("DB_HOST"), env)
    port = env.get("DB_PORT", "6543")
    name = env.get("DB_NAME", "postgres")
    user = env.get("DB_USER")
    password = env.get("DB_PASSWORD")

    if not all([host, user, password]):
        return None

    encoded_password = urllib.parse.quote_plus(password)
    db_url = f"postgresql://{user}:{encoded_password}@{host}:{port}/{name}"
    return normalize_db_url(db_url, env)

