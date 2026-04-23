from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

from config import request_timeout_seconds, user_agent, SourceDefinition


# result model

@dataclass
class SourceValidationResult:
    # Store the result of validating one source target. A target can be a whole source, a downloadable file, or a single web page.

    source_name: str
    target_web_address: str
    validation_scope: str
    is_allowed: bool
    failure_category: str | None
    reason: str
    permission_basis: str
    status_code: int | None = None
    robots_file_web_address: str | None = None
    robots_status_code: int | None = None
    robots_allowed: bool | None = None
    checked_at_universal_time: str = ""

    def to_dictionary(self) -> dict[str, Any]:
        # Return a plain dictionary version for json logs.
        return asdict(self)


# basic helpers

def get_current_timestamp_in_universal_time() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def build_robots_file_web_address(target_web_address: str) -> str:
    # Build the robots.txt address for a target page.
    parsed_web_address = urlparse(target_web_address)
    return f"{parsed_web_address.scheme}://{parsed_web_address.netloc}/robots.txt"


def create_default_request_session() -> requests.Session:
    # Create one shared request session. This keeps headers consistent across all source checks.

    request_session = requests.Session()
    request_session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "*/*",
        }
    )
    return request_session


# request failure classification

def classify_request_failure(request_exception: requests.RequestException) -> tuple[str, str]:
    # Turn a request exception into a simple failure category, a readable reason string.
    request_exception_text = str(request_exception)
    request_exception_text_lower = request_exception_text.lower()

    if isinstance(request_exception, requests.HTTPError):
        status_code = request_exception.response.status_code if request_exception.response is not None else None

        if status_code is not None:
            return "http_error", f"HTTP error {status_code}: {request_exception_text}"

        return "http_error", f"HTTP error: {request_exception_text}"

    if any(
        failure_fragment in request_exception_text_lower
        for failure_fragment in (
            "failed to resolve",
            "name resolution",
            "name or service not known",
            "nodename nor servname provided",
            "temporary failure in name resolution",
            "getaddrinfo failed",
        )
    ):
        return "dns_or_host_resolution_failure", f"DNS or host resolution failed: {request_exception_text}"

    if isinstance(request_exception, requests.Timeout):
        return "network_timeout", f"Network request timed out: {request_exception_text}"

    if isinstance(request_exception, requests.ConnectionError):
        return "network_connection_failure", f"Network connection failed: {request_exception_text}"

    return "network_request_failure", f"Request failed: {request_exception_text}"


# accessibility checks

def check_web_address_accessibility(
    request_session: requests.Session,
    target_web_address: str,
    timeout_seconds: int = request_timeout_seconds,
) -> tuple[bool, int | None, str, str | None]:
    # Check whether a web address is reachable. Returns: is_accessible, status_code, readable reason, and failure category if there was one.

    status_code = None

    try:
        response = request_session.get(
            target_web_address,
            timeout=timeout_seconds,
            allow_redirects=True,
            stream=True,
        )
        status_code = response.status_code
        response.close()

    except requests.RequestException as error:
        failure_category, reason = classify_request_failure(error)
        return False, None, reason, failure_category

    if status_code >= 400:
        return False, status_code, f"HTTP status {status_code}", "http_error"

    return True, status_code, "Web address is reachable.", None


def check_robots_permission(
    request_session: requests.Session,
    target_web_address: str,
    timeout_seconds: int = request_timeout_seconds,
) -> tuple[bool, str, int | None, bool | None, str | None]:
    # Check whether robots.txt allows automated access to a page. Returns: is_allowed, readable reason, robots status code, robots allowed value, ane failure category if there was one..

    robots_file_web_address = build_robots_file_web_address(target_web_address)
    robots_status_code = None

    try:
        response = request_session.get(
            robots_file_web_address,
            timeout=timeout_seconds,
            allow_redirects=True,
        )
        robots_status_code = response.status_code
        robots_text = response.text

    except requests.RequestException as error:
        failure_category, reason = classify_request_failure(error)
        return False, f"Could not fetch robots.txt: {reason}", None, None, failure_category

    # If robots.txt is missing or empty, we skip.
    if robots_status_code != 200 or not robots_text.strip():
        return False, "robots.txt was unavailable or empty, so the scraper skipped this page.", robots_status_code, None, "robots_file_unavailable"

    robot_file_parser = RobotFileParser()
    robot_file_parser.parse(robots_text.splitlines())

    # This asks robots.txt whether our declared user agent can fetch the target page.
    robots_allowed = robot_file_parser.can_fetch(user_agent, target_web_address)

    if not robots_allowed:
        return False, "robots.txt disallowed automated access for this page.", robots_status_code, False, "robots_disallowed"

    return True, "robots.txt allows automated access for this page.", robots_status_code, True, None


# source validation

def validate_source_definition(source_definition: SourceDefinition) -> SourceValidationResult:
    # Validate the source definition itself before touching the web. This checks things like enabled or disabled, or allowed or not allowed in config.

    checked_at_universal_time = get_current_timestamp_in_universal_time()

    if not source_definition.enabled:
        return SourceValidationResult(
            source_name=source_definition.source_name,
            target_web_address=source_definition.source_url,
            validation_scope="source",
            is_allowed=False,
            failure_category="source_disabled",
            reason="Source is disabled in config.py.",
            permission_basis=source_definition.permission_basis,
            checked_at_universal_time=checked_at_universal_time,
        )

    if source_definition.permission_status.lower() != "allowed":
        return SourceValidationResult(
            source_name=source_definition.source_name,
            target_web_address=source_definition.source_url,
            validation_scope="source",
            is_allowed=False,
            failure_category="source_permission_not_allowed",
            reason=f"Source permission status is '{source_definition.permission_status}', so it was skipped.",
            permission_basis=source_definition.permission_basis,
            checked_at_universal_time=checked_at_universal_time,
        )

    return SourceValidationResult(
        source_name=source_definition.source_name,
        target_web_address=source_definition.source_url,
        validation_scope="source",
        is_allowed=True,
        failure_category=None,
        reason="Source is marked as allowed in the manifest.",
        permission_basis=source_definition.permission_basis,
        checked_at_universal_time=checked_at_universal_time,
    )


def validate_download_source(
    request_session: requests.Session,
    source_definition: SourceDefinition,
) -> SourceValidationResult:
    # Validate a downloadable dataset source.
    # For downloads, we need source allowed in config nd target file endpoint reachable.

    base_result = validate_source_definition(source_definition)
    target_web_address = source_definition.downloadable_file_web_address or source_definition.source_url

    if not base_result.is_allowed:
        base_result.target_web_address = target_web_address
        return base_result

    is_accessible, status_code, reason, failure_category = check_web_address_accessibility(
        request_session=request_session,
        target_web_address=target_web_address,
    )

    return SourceValidationResult(
        source_name=source_definition.source_name,
        target_web_address=target_web_address,
        validation_scope="download",
        is_allowed=is_accessible,
        failure_category=failure_category,
        reason=reason,
        permission_basis=source_definition.permission_basis,
        status_code=status_code,
        checked_at_universal_time=get_current_timestamp_in_universal_time(),
    )


def validate_web_page_target(
    request_session: requests.Session,
    source_definition: SourceDefinition,
    page_web_address: str,
) -> SourceValidationResult:
    # Validate one web page target.
    # For page scraping, we may need: source allowed in config, robots.txt permission, and page reachability.

    base_result = validate_source_definition(source_definition)

    if not base_result.is_allowed:
        return SourceValidationResult(
            source_name=source_definition.source_name,
            target_web_address=page_web_address,
            validation_scope="page",
            is_allowed=False,
            failure_category=base_result.failure_category,
            reason=base_result.reason,
            permission_basis=source_definition.permission_basis,
            checked_at_universal_time=get_current_timestamp_in_universal_time(),
        )

    robots_file_web_address = None
    robots_status_code = None
    robots_allowed = None
    failure_category = None

    if source_definition.requires_robots_check:
        robots_file_web_address = build_robots_file_web_address(page_web_address)

        robots_check_allowed, robots_reason, robots_status_code, robots_allowed, failure_category = check_robots_permission(
            request_session=request_session,
            target_web_address=page_web_address,
        )

        if not robots_check_allowed:
            return SourceValidationResult(
                source_name=source_definition.source_name,
                target_web_address=page_web_address,
                validation_scope="page",
                is_allowed=False,
                failure_category=failure_category,
                reason=robots_reason,
                permission_basis=source_definition.permission_basis,
                robots_file_web_address=robots_file_web_address,
                robots_status_code=robots_status_code,
                robots_allowed=robots_allowed,
                checked_at_universal_time=get_current_timestamp_in_universal_time(),
            )

    is_accessible, status_code, accessibility_reason, failure_category = check_web_address_accessibility(
        request_session=request_session,
        target_web_address=page_web_address,
    )

    return SourceValidationResult(
        source_name=source_definition.source_name,
        target_web_address=page_web_address,
        validation_scope="page",
        is_allowed=is_accessible,
        failure_category=failure_category,
        reason=accessibility_reason,
        permission_basis=source_definition.permission_basis,
        status_code=status_code,
        robots_file_web_address=robots_file_web_address,
        robots_status_code=robots_status_code,
        robots_allowed=robots_allowed,
        checked_at_universal_time=get_current_timestamp_in_universal_time(),
    )