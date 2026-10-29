"""Safe, actionable service diagnostics without logging credentials or email bodies."""

import re


class InvalidServiceResponse(ValueError):
    """A service result failed its explicit output contract."""


def error_code(exc):
    children = getattr(exc, 'exceptions', ())
    if children:
        return error_code(children[0])
    name = type(exc).__name__.lower()
    status = (getattr(exc, 'status_code', None)
              or getattr(getattr(exc, 'response', None), 'status_code', None)
              or getattr(getattr(exc, 'resp', None), 'status', None))
    # The NVIDIA SDK raises a plain Exception with a leading "[HTTP status]".
    # Match only that documented prefix, never arbitrary numbers in error text.
    if status is None and (match := re.match(r'^\[(\d{3})\]', str(exc))):
        status = int(match.group(1))
    if isinstance(status, str) and status.isdigit():
        status = int(status)
    if status == 403 or 'permissiondenied' in name:
        return 'permission'
    if status == 401 or 'authentication' in name or 'refresherror' in name:
        return 'authentication'
    if status == 429 or 'ratelimit' in name:
        return 'rate_limit'
    if 'timeout' in name:
        return 'timeout'
    if 'programmingerror' in name or 'operationalerror' in name:
        return 'database'
    if isinstance(exc, InvalidServiceResponse) or name in {'validationerror', 'outputparserexception', 'interpretationresponseerror'}:
        return 'invalid_response'
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return 'internal_error'
    if 'connection' in name or isinstance(exc, OSError):
        return 'connection'
    return 'unavailable'


def error_hint(code, service='AI'):
    if code == 'authentication':
        return 'Reconnect Gmail to restore live search.' if service == 'Gmail' else 'The AI provider credentials need to be checked.'
    return {
        'permission': f'{service} denied access to the requested resource or model.',
        'rate_limit': f'{service} reached its request limit. Try again later.',
        'timeout': f'{service} did not respond within the time limit.',
        'invalid_response': f'{service} returned a response that could not be validated.',
        'internal_error': 'An internal request-processing error occurred.',
        'connection': f'Could not connect to {service}.',
        'database': 'The mailbox database needs attention before live search can finish.',
    }.get(code, f'{service} is currently unavailable.')
