"""Names invoked by framework dispatch or dataclass serialization, not direct calls."""

from local_ai_dictation.bridge import _BridgeHandler
from local_ai_dictation.types import DoctorIssue

# BaseHTTPRequestHandler dispatches HTTP methods and logging by attribute name.
_BridgeHandler.do_GET
_BridgeHandler.do_POST
_BridgeHandler.log_message

# dataclasses.asdict() includes this field in the doctor's JSON issue payloads.
DoctorIssue.remediation
