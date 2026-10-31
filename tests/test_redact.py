"""Redaction: one case per detector, Luhn gating, placeholder stability, custom patterns."""

from __future__ import annotations

import re

from omni.ingest.redact import Redactor, load_custom_patterns, luhn_ok, redact_request
from omni.ingest.schema import NormalizedRequest

# Secrets below are assembled at runtime so no secret-shaped literal is committed.
FAKE_AWS = "AKIA" + "Z" * 16
FAKE_OPENAI = "sk-" + "a1b2c3d4" * 3
FAKE_GH = "ghp_" + "x" * 36
FAKE_SLACK = "xoxb-" + "1234-5678-abcdefgh"
FAKE_JWT = ".".join(["ey" + "A" * 20, "ey" + "B" * 20, "C" * 20])


def _r(text: str) -> tuple[str, dict[str, int]]:
    r = Redactor()
    return r.text(text), dict(r.counts)


def test_email_phone_ipv4():
    out, counts = _r("mail jane.doe@example.com, call +1 415-555-0133, host 10.0.0.42 ok")
    assert out == "mail [EMAIL_1], call [PHONE_1], host [IPV4_1] ok"
    assert counts == {"EMAIL": 1, "PHONE": 1, "IPV4": 1}


def test_card_requires_luhn():
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")
    out, counts = _r("card 4111 1111 1111 1111 vs 4111-1111-1111-1112")
    assert out == "card [CARD_1] vs 4111-1111-1111-1112"
    assert counts == {"CARD": 1}


def test_api_key_shapes_bearer_and_jwt():
    text = f"keys {FAKE_AWS} {FAKE_OPENAI} {FAKE_GH} {FAKE_SLACK} sk-ant-not-a-real-key-000000"
    out, counts = _r(text)
    assert counts == {"API_KEY": 5}
    assert out == "keys [API_KEY_1] [API_KEY_2] [API_KEY_3] [API_KEY_4] [API_KEY_5]"
    out, counts = _r(f"Authorization: Bearer {FAKE_JWT}")
    assert out == "Authorization: [BEARER_1]" and counts == {"BEARER": 1}
    out, counts = _r(f"token={FAKE_JWT}")
    assert out == "token=[JWT_1]" and counts == {"JWT": 1}


def test_placeholders_stable_within_request_and_numbered_per_type():
    r = Redactor()
    a = r.text("a@x.io wrote to b@x.io then a@x.io again")
    assert a == "[EMAIL_1] wrote to [EMAIL_2] then [EMAIL_1] again"
    assert r.counts["EMAIL"] == 3
    # a new Redactor restarts numbering: stability is per request, not global
    assert Redactor().text("b@x.io") == "[EMAIL_1]"


def test_plain_numbers_and_versions_are_left_alone():
    out, counts = _r("order 48213 shipped 2026-10-30, v1.2.3, 12 items, total 1999.00")
    assert counts == {} and "48213" in out and "1999.00" in out


def test_custom_patterns(tmp_path, monkeypatch):
    f = tmp_path / "redaction.yaml"
    f.write_text('patterns:\n  - name: ticket id\n    regex: "TKT-\\\\d{6}"\n', encoding="utf-8")
    monkeypatch.setenv("OMNI_REDACTION_FILE", str(f))
    custom = load_custom_patterns()
    assert custom and custom[0][0] == "TICKET_ID" and isinstance(custom[0][1], re.Pattern)
    r = Redactor(custom)
    assert r.text("see TKT-004711 and TKT-004711") == "see [TICKET_ID_1] and [TICKET_ID_1]"
    monkeypatch.setenv("OMNI_REDACTION_FILE", str(tmp_path / "missing.yaml"))
    assert load_custom_patterns() == []


def test_redact_request_covers_messages_tool_args_and_response():
    rec = NormalizedRequest.model_validate(
        {
            "request_id": "r1",
            "ts": "2026-10-30T00:00:00Z",
            "messages": [
                {"role": "user", "content": "I am jane@x.io"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c", "name": "save", "arguments": {"to": "jane@x.io", "n": 3}}
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c",
                    "content": [{"type": "text", "text": "sent to jane@x.io"}],
                },
            ],
            "response": {
                "content": None,
                "tool_calls": [{"name": "k", "arguments": f'{{"v": "{FAKE_GH}"}}'}],
            },
        }
    )
    red, counts = redact_request(rec)
    assert red.messages[0].content == "I am [EMAIL_1]"
    assert red.messages[1].tool_calls[0].arguments == {"to": "[EMAIL_1]", "n": 3}
    assert red.messages[2].content == [{"type": "text", "text": "sent to [EMAIL_1]"}]
    assert red.response.tool_calls[0].arguments == '{"v": "[API_KEY_1]"}'
    assert counts == {"EMAIL": 3, "API_KEY": 1}
    assert rec.messages[0].content == "I am jane@x.io"  # original untouched
    assert red.raw_hash != rec.raw_hash
