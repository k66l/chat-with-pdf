"""FastAPI middleware for request/response logging."""

import time
import json
from typing import Callable, Dict, Any
import structlog
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Message
import uuid


logger = structlog.get_logger(__name__)


class RequestResponseLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all HTTP requests and responses."""

    def __init__(self, app):
        super().__init__(app)
        self.excluded_paths = {"/docs", "/redoc",
                               "/openapi.json", "/favicon.ico"}
        self.sensitive_fields = {"password", "token", "api_key", "secret"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and response with comprehensive logging."""
        # Generate unique request ID
        request_id = str(uuid.uuid4())[:8]

        # Start timing
        start_time = time.time()

        # Skip logging for documentation and static files
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        # Log request
        await self._log_request(request, request_id)

        # Process request and capture response
        try:
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Log response
            await self._log_response(request, response, request_id, process_time)

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.4f}"

            return response

        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            await self._log_error(request, e, request_id, process_time)
            raise

    async def _log_request(self, request: Request, request_id: str) -> None:
        """Log incoming request details."""
        try:
            # Get client information
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("user-agent", "Unknown")

            # Read request body if present
            request_body = None
            if request.method in ["POST", "PUT", "PATCH"]:
                request_body = await self._get_request_body(request)

            # Prepare request log data
            request_data = {
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_ip": client_ip,
                "user_agent": user_agent,
                "headers": self._sanitize_headers(dict(request.headers)),
                "body": self._sanitize_body(request_body),
                "content_type": request.headers.get("content-type"),
                "content_length": request.headers.get("content-length")
            }

            logger.info(
                "HTTP Request",
                **request_data,
                event_type="request"
            )

        except Exception as e:
            logger.error("Error logging request",
                         error=str(e), request_id=request_id)

    async def _log_response(
        self,
        request: Request,
        response: Response,
        request_id: str,
        process_time: float
    ) -> None:
        """Log outgoing response details."""
        try:
            # Get response body for logging
            response_body = None
            if hasattr(response, 'body') and response.body:
                try:
                    response_body = response.body.decode('utf-8')
                    # Try to parse as JSON for better logging
                    if response.headers.get("content-type") == "application/json":
                        response_body = json.loads(response_body)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    response_body = f"<binary data: {len(response.body)} bytes>"

            # Prepare response log data
            response_data = {
                "request_id": request_id,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": self._sanitize_response_body(response_body),
                "process_time_seconds": round(process_time, 4),
                "method": request.method,
                "path": request.url.path
            }

            # Determine log level based on status code
            if response.status_code >= 500:
                log_level = "error"
            elif response.status_code >= 400:
                log_level = "warning"
            else:
                log_level = "info"

            getattr(logger, log_level)(
                "HTTP Response",
                **response_data,
                event_type="response"
            )

        except Exception as e:
            logger.error("Error logging response",
                         error=str(e), request_id=request_id)

    async def _log_error(
        self,
        request: Request,
        error: Exception,
        request_id: str,
        process_time: float
    ) -> None:
        """Log request processing errors."""
        try:
            error_data = {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "process_time_seconds": round(process_time, 4)
            }

            logger.error(
                "HTTP Request Error",
                **error_data,
                event_type="error"
            )

        except Exception as e:
            logger.error("Error logging error",
                         error=str(e), request_id=request_id)

    async def _get_request_body(self, request: Request) -> Any:
        """Safely extract request body."""
        try:
            body = await request.body()
            if not body:
                return None

            # Try to decode as JSON first
            try:
                return json.loads(body.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # If not JSON, return as string if possible
                try:
                    return body.decode('utf-8')
                except UnicodeDecodeError:
                    return f"<binary data: {len(body)} bytes>"

        except Exception:
            return None

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first (proxy/load balancer)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fallback to direct client IP
        if request.client:
            return request.client.host

        return "unknown"

    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Remove sensitive information from headers."""
        sanitized = {}
        for key, value in headers.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in self.sensitive_fields):
                sanitized[key] = "***REDACTED***"
            elif key_lower in ["authorization", "cookie"]:
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value
        return sanitized

    def _sanitize_body(self, body: Any) -> Any:
        """Remove sensitive information from request body."""
        if not body:
            return body

        if isinstance(body, dict):
            sanitized = {}
            for key, value in body.items():
                key_lower = key.lower()
                if any(sensitive in key_lower for sensitive in self.sensitive_fields):
                    sanitized[key] = "***REDACTED***"
                else:
                    sanitized[key] = value
            return sanitized

        return body

    def _sanitize_response_body(self, body: Any) -> Any:
        """Sanitize response body for logging."""
        if not body:
            return body

        # For large responses, truncate for logging
        if isinstance(body, str) and len(body) > 5000:
            return body[:5000] + "... <truncated>"

        if isinstance(body, dict):
            # Don't log extremely large response bodies
            body_str = json.dumps(body)
            if len(body_str) > 5000:
                return {"message": "Response body truncated for logging", "size": len(body_str)}

        return body


# Global metrics collector
class MetricsCollector:
    """Global metrics collector for application statistics."""

    def __init__(self):
        self.request_count = 0
        self.total_response_time = 0.0
        self.status_codes = {}
        self.endpoint_metrics = {}

    def update_metrics(self, request: Request, response: Response, process_time: float):
        """Update metrics for a completed request."""
        try:
            self.request_count += 1
            self.total_response_time += process_time

            # Track status codes
            status_code = response.status_code
            self.status_codes[status_code] = self.status_codes.get(
                status_code, 0) + 1

            # Track endpoint metrics
            endpoint = f"{request.method} {request.url.path}"
            if endpoint not in self.endpoint_metrics:
                self.endpoint_metrics[endpoint] = {
                    "count": 0,
                    "total_time": 0.0,
                    "avg_time": 0.0
                }

            self.endpoint_metrics[endpoint]["count"] += 1
            self.endpoint_metrics[endpoint]["total_time"] += process_time
            self.endpoint_metrics[endpoint]["avg_time"] = (
                self.endpoint_metrics[endpoint]["total_time"] /
                self.endpoint_metrics[endpoint]["count"]
            )

            # Log metrics periodically (every 10 requests)
            if self.request_count % 10 == 0:
                avg_response_time = self.total_response_time / self.request_count

                logger.info(
                    "Request Metrics Summary",
                    total_requests=self.request_count,
                    avg_response_time=round(avg_response_time, 4),
                    status_codes=self.status_codes,
                    top_endpoints=dict(
                        list(self.endpoint_metrics.items())[:5]),
                    event_type="metrics"
                )

        except Exception as e:
            logger.error("Error updating metrics", error=str(e))

    def update_error_metrics(self, request: Request, process_time: float):
        """Update metrics for failed requests."""
        try:
            self.request_count += 1
            self.total_response_time += process_time

            # Count as 500 error
            self.status_codes[500] = self.status_codes.get(500, 0) + 1

        except Exception as e:
            logger.error("Error updating error metrics", error=str(e))

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        avg_response_time = 0.0
        if self.request_count > 0:
            avg_response_time = self.total_response_time / self.request_count

        return {
            "total_requests": self.request_count,
            "average_response_time": round(avg_response_time, 4),
            "status_codes": self.status_codes,
            "endpoint_metrics": self.endpoint_metrics
        }


# Global instance
global_metrics = MetricsCollector()


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect request metrics and statistics."""

    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Collect metrics for each request."""
        start_time = time.time()

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # Update global metrics
            global_metrics.update_metrics(request, response, process_time)

            return response

        except Exception as e:
            process_time = time.time() - start_time
            # Log error metrics
            global_metrics.update_error_metrics(request, process_time)
            raise
