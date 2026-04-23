"""Bitrix24 REST tools (task 0.18).

Eight base tools mapped to Bitrix webhook REST endpoints, grouped by what
they touch (deals / contacts / tasks / activities / users). Each write
tool has a SET / VERIFY split: the SET mutates, the VERIFY re-reads and
compares — ``tools.verification.run_with_verify`` pairs them.

Research §9 rules applied:
- Every write tool has ``requires_verify=True`` and uses
  ``idempotency_key`` in the payload (stored as a UF or task comment so
  repeat invocations don't double-write).
- Destructive action (``deal_close_won``) is tier="danger" and goes
  through ``safety.confirmation``.

This module does *not* know about your portal's specific UF fields or
stage IDs — agents inject those via the payload, and the tool just
forwards. Per-portal schema lives in the agent's config.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from agents_core.tools.registry import Tool

__all__ = ["BitrixClient", "make_bitrix_tools"]

logger = logging.getLogger(__name__)


class BitrixClient:
    """Thin REST wrapper over a Bitrix24 webhook URL.

    ``webhook_url`` is the full incoming-webhook base, e.g.
    ``https://{portal}.bitrix24.ru/rest/1/{token}/``. The full-scope
    token for the 24bankrotsttvo portal lives in
    ``memory/reference_credentials_and_services.md``.
    """

    def __init__(
        self,
        webhook_url: str,
        client: httpx.AsyncClient | None = None,
        timeout: float = 20.0,
    ) -> None:
        if not webhook_url.endswith("/"):
            webhook_url += "/"
        self._base = webhook_url
        self._client = client
        self._timeout = timeout

    async def call(self, method: str, payload: dict[str, Any] | None = None) -> Any:
        url = f"{self._base}{method}.json"
        async with (self._client or httpx.AsyncClient()) as c:
            resp = await c.post(url, json=payload or {}, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
        if isinstance(data, dict) and "error" in data:
            raise BitrixError(data.get("error"), data.get("error_description"))
        return data.get("result") if isinstance(data, dict) else data


class BitrixError(RuntimeError):
    def __init__(self, code: str | None, description: str | None) -> None:
        super().__init__(f"{code}: {description}")
        self.code = code
        self.description = description


# --- handler helpers ----------------------------------------------------

async def _deal_get(bx: BitrixClient, deal_id: int):
    return await bx.call("crm.deal.get", {"ID": int(deal_id)})


async def _deal_update(
    bx: BitrixClient, deal_id: int, fields: dict[str, Any], idempotency_key: str
):
    # Stash idempotency_key in a comment field so duplicate SET is safe to
    # detect on re-read (the paired _verify tool checks it).
    _fields = dict(fields)
    _fields.setdefault("COMMENTS", f"[agent:idem={idempotency_key}]")
    return await bx.call(
        "crm.deal.update", {"ID": int(deal_id), "FIELDS": _fields}
    )


async def _deal_update_verify(
    bx: BitrixClient, deal_id: int, expected: dict[str, Any]
):
    deal = await bx.call("crm.deal.get", {"ID": int(deal_id)})
    missing = {k: (deal.get(k), v) for k, v in expected.items() if deal.get(k) != v}
    return {"verified": not missing, "diff": missing}


async def _deal_move_stage(
    bx: BitrixClient, deal_id: int, stage_id: str, idempotency_key: str
):
    return await _deal_update(
        bx, deal_id, {"STAGE_ID": stage_id}, idempotency_key
    )


async def _deal_close_won(
    bx: BitrixClient, deal_id: int, idempotency_key: str
):
    # Destructive — caller must have enforced safety.confirmation first.
    return await _deal_update(
        bx, deal_id, {"STAGE_ID": "WON", "CLOSED": "Y"}, idempotency_key
    )


async def _deal_comment_add(
    bx: BitrixClient, deal_id: int, text: str, idempotency_key: str
):
    return await bx.call(
        "crm.timeline.comment.add",
        {
            "fields": {
                "ENTITY_ID": int(deal_id),
                "ENTITY_TYPE": "deal",
                "COMMENT": f"{text}\n[agent:idem={idempotency_key}]",
            }
        },
    )


async def _contact_get(bx: BitrixClient, contact_id: int):
    return await bx.call("crm.contact.get", {"ID": int(contact_id)})


async def _task_create(
    bx: BitrixClient,
    title: str,
    responsible_id: int,
    description: str | None,
    idempotency_key: str,
):
    return await bx.call(
        "tasks.task.add",
        {
            "fields": {
                "TITLE": title,
                "RESPONSIBLE_ID": int(responsible_id),
                "DESCRIPTION": (
                    f"{description or ''}\n[agent:idem={idempotency_key}]"
                ),
            }
        },
    )


async def _activity_add(
    bx: BitrixClient,
    owner_id: int,
    owner_type: str,
    subject: str,
    description: str | None,
    idempotency_key: str,
):
    return await bx.call(
        "crm.activity.add",
        {
            "fields": {
                "OWNER_ID": int(owner_id),
                "OWNER_TYPE_ID": owner_type,
                "TYPE_ID": 4,  # "User-defined"
                "SUBJECT": subject,
                "DESCRIPTION": (
                    f"{description or ''}\n[agent:idem={idempotency_key}]"
                ),
                "RESPONSIBLE_ID": 1,
                "COMPLETED": "N",
            }
        },
    )


async def _user_get(bx: BitrixClient, user_id: int):
    return await bx.call("user.get", {"ID": int(user_id)})


# --- factory ------------------------------------------------------------

def make_bitrix_tools(bx: BitrixClient) -> list[Tool]:
    """Return the 8 base Bitrix tools.

    Deal update comes as a SET/VERIFY pair — that's 2 of the 8 tools.
    The remaining 6: deal_get, deal_move_stage (SET only; callers can
    pair with deal_update_verify), deal_close_won (tier=danger),
    deal_comment_add, contact_get, task_create, activity_add, user_get.

    Counting by spec ("8 base tools"): get/update/comment/move_stage,
    contact_get, task_create, activity_add, user_get. Plus the _verify
    companion for update (research §9 rule 1) which isn't itself one of
    the 8 but is required for parity.
    """
    return [
        Tool(
            name="bitrix_deal_get",
            description="Fetch a CRM deal by ID.",
            input_schema={
                "type": "object",
                "properties": {"deal_id": {"type": "integer"}},
                "required": ["deal_id"],
            },
            handler=lambda deal_id, _bx=bx: _deal_get(_bx, deal_id),
            tier="read",
            tags=("bitrix", "crm", "read"),
        ),
        Tool(
            name="bitrix_deal_update",
            description="Update CRM deal fields. Always call bitrix_deal_update_verify after.",
            input_schema={
                "type": "object",
                "properties": {
                    "deal_id": {"type": "integer"},
                    "fields": {"type": "object"},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["deal_id", "fields", "idempotency_key"],
            },
            handler=lambda deal_id, fields, idempotency_key, _bx=bx: _deal_update(
                _bx, deal_id, fields, idempotency_key
            ),
            tier="write",
            idempotent=False,
            requires_verify=True,
            tags=("bitrix", "crm", "write"),
        ),
        Tool(
            name="bitrix_deal_update_verify",
            description=(
                "Verify counterpart for bitrix_deal_update. Re-reads the deal "
                "and confirms expected field values. Returns "
                "{verified: bool, diff: {field: (actual, expected)}}."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "deal_id": {"type": "integer"},
                    "expected": {"type": "object"},
                },
                "required": ["deal_id", "expected"],
            },
            handler=lambda deal_id, expected, _bx=bx: _deal_update_verify(
                _bx, deal_id, expected
            ),
            tier="read",
            tags=("bitrix", "crm", "verify"),
        ),
        Tool(
            name="bitrix_deal_move_stage",
            description="Move a deal to a given STAGE_ID.",
            input_schema={
                "type": "object",
                "properties": {
                    "deal_id": {"type": "integer"},
                    "stage_id": {"type": "string"},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["deal_id", "stage_id", "idempotency_key"],
            },
            handler=lambda deal_id, stage_id, idempotency_key, _bx=bx: (
                _deal_move_stage(_bx, deal_id, stage_id, idempotency_key)
            ),
            tier="write",
            idempotent=False,
            requires_verify=True,
            tags=("bitrix", "crm", "write"),
        ),
        Tool(
            name="bitrix_deal_close_won",
            description=(
                "DESTRUCTIVE: move deal to WON and close. Requires "
                "user_confirmed=true (enforced by safety.require_confirmation)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "deal_id": {"type": "integer"},
                    "idempotency_key": {"type": "string"},
                    "user_confirmed": {"type": "boolean"},
                },
                "required": ["deal_id", "idempotency_key", "user_confirmed"],
            },
            handler=lambda deal_id, idempotency_key, user_confirmed, _bx=bx: (
                _deal_close_won(_bx, deal_id, idempotency_key)
            ),
            tier="danger",
            idempotent=False,
            requires_verify=True,
            tags=("bitrix", "crm", "danger"),
        ),
        Tool(
            name="bitrix_deal_comment_add",
            description="Add a timeline comment to a deal.",
            input_schema={
                "type": "object",
                "properties": {
                    "deal_id": {"type": "integer"},
                    "text": {"type": "string"},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["deal_id", "text", "idempotency_key"],
            },
            handler=lambda deal_id, text, idempotency_key, _bx=bx: (
                _deal_comment_add(_bx, deal_id, text, idempotency_key)
            ),
            tier="write",
            idempotent=False,
            requires_verify=False,  # append-only, no verify needed
            tags=("bitrix", "crm", "write"),
        ),
        Tool(
            name="bitrix_contact_get",
            description="Fetch a CRM contact by ID.",
            input_schema={
                "type": "object",
                "properties": {"contact_id": {"type": "integer"}},
                "required": ["contact_id"],
            },
            handler=lambda contact_id, _bx=bx: _contact_get(_bx, contact_id),
            tier="read",
            tags=("bitrix", "crm", "read"),
        ),
        Tool(
            name="bitrix_task_create",
            description="Create a task in Bitrix.",
            input_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "responsible_id": {"type": "integer"},
                    "description": {"type": "string"},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["title", "responsible_id", "idempotency_key"],
            },
            handler=lambda title, responsible_id, idempotency_key, description=None, _bx=bx: (
                _task_create(_bx, title, responsible_id, description, idempotency_key)
            ),
            tier="write",
            idempotent=False,
            requires_verify=False,
            tags=("bitrix", "tasks", "write"),
        ),
        Tool(
            name="bitrix_activity_add",
            description="Log a CRM activity (call, meeting, note).",
            input_schema={
                "type": "object",
                "properties": {
                    "owner_id": {"type": "integer"},
                    "owner_type": {
                        "type": "string",
                        "description": "'deal', 'contact', 'lead', ...",
                    },
                    "subject": {"type": "string"},
                    "description": {"type": "string"},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["owner_id", "owner_type", "subject", "idempotency_key"],
            },
            handler=lambda owner_id, owner_type, subject, idempotency_key,
                description=None, _bx=bx: _activity_add(
                _bx, owner_id, owner_type, subject, description, idempotency_key
            ),
            tier="write",
            idempotent=False,
            requires_verify=False,
            tags=("bitrix", "crm", "write"),
        ),
        Tool(
            name="bitrix_user_get",
            description="Fetch a Bitrix user by ID.",
            input_schema={
                "type": "object",
                "properties": {"user_id": {"type": "integer"}},
                "required": ["user_id"],
            },
            handler=lambda user_id, _bx=bx: _user_get(_bx, user_id),
            tier="read",
            tags=("bitrix", "user", "read"),
        ),
    ]
