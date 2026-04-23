"""HH.ru API tools (task 0.22).

Three base tools: ``hh_get_resumes``, ``hh_get_applications``,
``hh_send_invitation``. OAuth 2.0 token is injected at construction —
refresh is the caller's concern; the tool just uses what it's given.

HH.ru gotchas (from memory/hh_api_credentials.md):
- All endpoints require ``HH-User-Agent: AppName (email)`` header.
- Pagination via ``page`` / ``per_page`` (max 100).
- Rate limited ~= 200 req/minute; pair with ``safety.rate_limits`` on
  the ``tier="read"`` bucket in busy agents.
"""

from __future__ import annotations

from typing import Any

import httpx

from agents_core.tools.registry import Tool

__all__ = ["HHClient", "make_hh_tools"]

_BASE_URL = "https://api.hh.ru"


class HHClient:
    def __init__(
        self,
        access_token: str,
        user_agent: str,
        client: httpx.AsyncClient | None = None,
        timeout: float = 20.0,
    ) -> None:
        if not access_token:
            raise ValueError("access_token required")
        if not user_agent:
            raise ValueError(
                "user_agent required (HH-User-Agent header mandatory on API)"
            )
        self._headers = {
            "Authorization": f"Bearer {access_token}",
            "HH-User-Agent": user_agent,
        }
        self._client = client
        self._timeout = timeout

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{_BASE_URL}{path}"
        async with (self._client or httpx.AsyncClient()) as c:
            resp = await c.request(
                method,
                url,
                headers=self._headers,
                params=params,
                json=json,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            if resp.status_code == 204 or not resp.content:
                return {}
            return resp.json()


def make_hh_tools(hh: HHClient) -> list[Tool]:
    async def _resumes(
        text: str | None = None,
        area: int | None = None,
        per_page: int = 20,
        page: int = 0,
    ):
        params = {"per_page": int(per_page), "page": int(page)}
        if text:
            params["text"] = text
        if area is not None:
            params["area"] = int(area)
        return await hh._request("GET", "/resumes", params=params)

    async def _applications(vacancy_id: int, per_page: int = 20, page: int = 0):
        return await hh._request(
            "GET",
            "/negotiations",
            params={
                "vacancy_id": int(vacancy_id),
                "per_page": int(per_page),
                "page": int(page),
            },
        )

    async def _invitation(
        resume_id: str,
        vacancy_id: int,
        message: str,
        idempotency_key: str,
    ):
        # HH's invitation endpoint: POST /negotiations/phone_interview/{resume}
        # is flaky in practice; the canonical route is POST /negotiations with
        # ``resume_id`` + ``vacancy_id`` + ``message``. ``idempotency_key`` is
        # prepended to ``message`` so duplicate POSTs look identical to an
        # operator auditing the HH UI.
        payload = {
            "resume_id": str(resume_id),
            "vacancy_id": int(vacancy_id),
            "message": f"{message}\n\n[idem:{idempotency_key}]",
        }
        return await hh._request("POST", "/negotiations", json=payload)

    return [
        Tool(
            name="hh_get_resumes",
            description=(
                "Search HH.ru resumes. Returns paged response with 'items'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "area": {"type": "integer", "description": "HH area id (e.g. 88 = Казань)"},
                    "per_page": {"type": "integer", "default": 20, "maximum": 100},
                    "page": {"type": "integer", "default": 0},
                },
            },
            handler=_resumes,
            tier="read",
            tags=("hh", "read"),
        ),
        Tool(
            name="hh_get_applications",
            description="List applications (negotiations) for a vacancy.",
            input_schema={
                "type": "object",
                "properties": {
                    "vacancy_id": {"type": "integer"},
                    "per_page": {"type": "integer", "default": 20, "maximum": 100},
                    "page": {"type": "integer", "default": 0},
                },
                "required": ["vacancy_id"],
            },
            handler=_applications,
            tier="read",
            tags=("hh", "read"),
        ),
        Tool(
            name="hh_send_invitation",
            description="Invite a candidate by resume_id to vacancy_id with a message.",
            input_schema={
                "type": "object",
                "properties": {
                    "resume_id": {"type": "string"},
                    "vacancy_id": {"type": "integer"},
                    "message": {"type": "string"},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["resume_id", "vacancy_id", "message", "idempotency_key"],
            },
            handler=_invitation,
            tier="write",
            idempotent=False,
            requires_verify=False,
            tags=("hh", "write"),
        ),
    ]
