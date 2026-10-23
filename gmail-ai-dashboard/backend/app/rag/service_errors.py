"""Safe, actionable service diagnostics without logging credentials or email bodies."""


def error_code(exc):
    children = getattr(exc, 'exceptions', ())
    if children:
        return error_code(children[0])
    name = type(exc).__name__.lower()
    status = getattr(exc, 'status_code', None) or getattr(getattr(exc, 'resp', None), 'status', None)
    if status in (401, 403) or 'authentication' in name or 'refresherror' in name:
        return 'authentication'
    if status == 429 or 'ratelimit' in name:
        return 'rate_limit'
    if 'timeout' in name:
        return 'timeout'
    if 'programmingerror' in name or 'operationalerror' in name:
        return 'database'
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return 'invalid_response'
    if 'connection' in name or isinstance(exc, OSError):
        return 'connection'
    return 'unavailable'


def error_hint(code, service='AI'):
    if code == 'authentication':
        return 'Reconnect Gmail to restore live search.' if service == 'Gmail' else 'The AI provider credentials need to be checked.'
    return {
        'rate_limit': f'{service} reached its request limit. Try again later.',
        'timeout': f'{service} did not respond within the time limit.',
        'invalid_response': f'{service} returned a response that could not be validated.',
        'connection': f'Could not connect to {service}.',
        'database': 'The mailbox database needs attention before live search can finish.',
    }.get(code, f'{service} is currently unavailable.')
