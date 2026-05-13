from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

from config import request_timeout_seconds, user_agent, SourceDefinition


#Result model --------------------------------------


@dataclass
class SourceValidationResult:
    # Store the result of validating one source target.
    # A target can be:
    # - a whole source
    # - a downloadable file
    # - a single web page
    source_name: str
    url: str
    validation_scope: str
    is_allowed: bool
    fail_type: str | None
    reason: str
    permission_basis: str
    status_code: int | None = None
    robots_file_web_address: str | None = None
    robots_status_code: int | None = None
    robots_allowed: bool | None = None
    checked_at_universal_time: str = ""

    def to_dictionary(self) -> dict[str, Any]:
        #Return a plain dictionary version for json logs.
        return asdict(self)


#Basic helpers --------------------------------------


def get_current_timestamp_in_universal_time() -> str:
    #Return the current UTC timestamp.
    return datetime.now(timezone.utc).isoformat()


def build_robots_file_web_address(target_web_address: str) -> str:
    #Build the robots.txt address for a target page.
    parsed_web_address = urlparse(url)
    return f"{parsed_web_address.scheme}://{parsed_web_address.netloc}/robots.txt"


def create_default_request_session() -> requests.Session:
    #Create one shared request session.
    #This keeps headers consistent across all source checks.
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "*/*",
        }
    )
    return session


#Request failure classification --------------------------------------


def classify_request_failure(request_exception: requests.RequestException) -> tuple[str, str]:
    # Turn a request exception into:
    # - a simple failure category
    # - a readable reason string
    request_exception_text = str(request_exception)
    request_exception_text_lower = request_exception_text.lower()

    if isinstance(request_exception, requests.HTTPError):
        status_code = request_exception.response.status_code if request_exception.response is not None else None

        if status_code is not None:
            return "http_error", f"HTTP error {status_code}: {request_exception_text}"

        return "http_error", f"HTTP error: {request_exception_text}"

    if any(
        fail_part in request_exception_text_lower
        for fail_part in (
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


#Accessibility checks --------------------------------------


def check_web_address_accessibility(
    session: requests.Session,
    url: str,
    timeout_seconds: int = request_timeout_seconds,
) -> tuple[bool, int | None, str, str | None]:
    # Check whether a web address is reachable.
    # Returns:
    # - is_accessible
    # - status_code
    # - readable reason
    # - failure category if there was one
    status_code = None

    try:
        response = session.get(
            url,
            timeout=timeout_seconds,
            allow_redirects=True,
            stream=True,
        )
        status_code = response.status_code
        response.close()

    except requests.RequestException as error:
        fail_type, reason = classify_request_failure(error)
        return False, None, reason, fail_type

    if status_code >= 400:
        return False, status_code, f"HTTP status {status_code}", "http_error"

    return True, status_code, "Web address is reachable.", None


def check_robots_permission(
    session: requests.Session,
    url: str,
    timeout_seconds: int = request_timeout_seconds,
) -> tuple[bool, str, int | None, bool | None, str | None]:
    # Check whether robots.txt allows automated access to a page.
    # Returns:
    # - is_allowed
    # - readable reason
    # - robots status code
    # - robots allowed value
    # - failure category if there was one
    robots_file_web_address = build_robots_file_web_address(url)
    robots_status_code = None

    try:
        response = session.get(
            url,
            timeout=timeout_seconds,
            allow_redirects=True,
        )
        robots_status_code = response.status_code
        robots_text = response.text

    except requests.RequestException as error:
        fail_type, reason = classify_request_failure(error)
        return False, f"Could not fetch robots.txt: {reason}", None, None, fail_type

    #If robots.txt is missing or empty, stay conservative and skip.
    if robots_status_code != 200 or not robots_text.strip():
        return False, "robots.txt was unavailable or empty, so the scraper skipped this page.", robots_status_code, None, "robots_file_unavailable"

    robot_file_parser = RobotFileParser()
    robot_file_parser.parse(robots_text.splitlines())

    # This checks whether our declared user agent can fetch the target page.
    robots_allowed = robot_file_parser.can_fetch(user_agent, url)

    if not robots_allowed:
        return False, "robots.txt disallowed automated access for this page.", robots_status_code, False, "robots_disallowed"

    return True, "robots.txt allows automated access for this page.", robots_status_code, True, None


#Source validation --------------------------------------

def validate_source_definition(source: SourceDefinition) -> SourceValidationResult:
    #Validate the source definition itself before touching the web.
    # This checks things like:
    # - enabled or disabled
    # - allowed or not allowed in config
    checked_at_universal_time = get_current_timestamp_in_universal_time()

    if not source.enabled:
        return SourceValidationResult(
            source_name=source.source_name,
            url=source.source_url,
            validation_scope="source",
            is_allowed=False,
            fail_type="source_disabled",
            reason="Source is disabled in config.py.",
            permission_basis=source.permission_basis,
            checked_at_universal_time=checked_at_universal_time,
        )

    if source.permission_status.lower() != "allowed":
        return SourceValidationResult(
            source_name=source.source_name,
            url=source.source_url,
            validation_scope="source",
            is_allowed=False,
            fail_type="source_permission_not_allowed",
            reason=f"Source permission status is '{source.permission_status}', so it was skipped.",
            permission_basis=source.permission_basis,
            checked_at_universal_time=checked_at_universal_time,
        )

    return SourceValidationResult(
        source_name=source.source_name,
        url=source.source_url,
        validation_scope="source",
        is_allowed=True,
        fail_type=None,
        reason="Source is marked as allowed in the manifest.",
        permission_basis=source.permission_basis,
        checked_at_universal_time=checked_at_universal_time,
    )


def validate_download_source(
    session: requests.Session,
    source: SourceDefinition,
) -> SourceValidationResult:
    # Validate a downloadable dataset source.
    # For downloads, we need:
    # - source allowed in config
    # - target file endpoint reachable
    base_result = validate_source_definition(source)
    url = source.downloadable_file_web_address or source.source_url

    if not base_result.is_allowed:
        base_result.url = url
        return base_result

    is_accessible, status_code, reason, fail_type = check_web_address_accessibility(
        session=session,
        url=url,
    )

    return SourceValidationResult(
        source_name=source.source_name,
        url=url,
        validation_scope="download",
        is_allowed=is_accessible,
        fail_type=fail_type,
        reason=reason,
        permission_basis=source.permission_basis,
        status_code=status_code,
        checked_at_universal_time=get_current_timestamp_in_universal_time(),
    )


def validate_web_page_target(
    session: requests.Session,
    source: SourceDefinition,
    url: str,
) -> SourceValidationResult:
    # Validate one web page target.
    # For page scraping, we may need:
    # - source allowed in config
    # - robots.txt permission
    # - page reachability
    base_result = validate_source_definition(source)

    if not base_result.is_allowed:
        return SourceValidationResult(
            source_name=source.source_name,
            url=url,
            validation_scope="page",
            is_allowed=False,
            fail_type=base_result.fail_type,
            reason=base_result.reason,
            permission_basis=source.permission_basis,
            checked_at_universal_time=get_current_timestamp_in_universal_time(),
        )

    robots_file_web_address = None
    robots_status_code = None
    robots_allowed = None
    fail_type = None

    if source.requires_robots_check:
        robots_file_web_address = build_robots_file_web_address(page_url)

        robots_check_allowed, robots_reason, robots_status_code, robots_allowed, fai_type = check_robots_permission(
            session=session,
            url=page_url
        )

        if not robots_check_allowed:
            return SourceValidationResult(
                source_name=source.source_name,
                url=page_url,
                validation_scope="page",
                is_allowed=False,
                fail_type=fail_type,
                reason=robots_reason,
                permission_basis=source.permission_basis,
                robots_file_web_address=robots_file_web_address,
                robots_status_code=robots_status_code,
                robots_allowed=robots_allowed,
                checked_at_universal_time=get_current_timestamp_in_universal_time(),
            )

    is_accessible, status_code, accessibility_reason, fail_type = check_web_address_accessibility(
        session=session,
        url=page_url,
    )

    return SourceValidationResult(
        source_name=source.source_name,
        url=page_url,
        validation_scope="page",
        is_allowed=is_accessible,
        fail_type=fail_type
        reason=accessibility_reason,
        permission_basis=source.permission_basis,
        status_code=status_code,
        robots_file_web_address=robots_file_web_address,
        robots_status_code=robots_status_code,
        robots_allowed=robots_allowed,
        checked_at_universal_time=get_current_timestamp_in_universal_time(),
    )
