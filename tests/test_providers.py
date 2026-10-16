import json

import httpx
import pytest

from sourcebook.models import Source
from sourcebook.providers import (
    HTTPProvider,
    ProviderError,
    SpanProvider,
    configured_provider,
    split_spans,
)


@pytest.fixture
def sources():
    return [
        Source(
            id="s1",
            document_id="d1",
            name="example.txt",
            page=1,
            section="",
            start=0,
            end=30,
            text="Evidence requires a source.",
        )
    ]


def mock_client(monkeypatch, handler):
    real = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs)
    )


@pytest.mark.parametrize("kind", ["ollama", "compatible"])
def test_provider_contract(monkeypatch, sources, kind):
    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["messages"][0]["role"] == "system"
        assert "untrusted" in body["messages"][0]["content"]
        assert request.headers["Authorization"] == "Bearer test-key"
        content = json.dumps(
            {
                "claims": [
                    {
                        "text": "A source is required.",
                        "evidence": [{"source_id": "s1", "quote": sources[0].text}],
                    }
                ]
            }
        )
        if kind == "ollama":
            assert request.url.path == "/api/chat"
            assert body["format"]["type"] == "object"
            return httpx.Response(200, json={"message": {"content": content}})
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    mock_client(monkeypatch, handler)
    base = "http://localhost:11434" if kind == "ollama" else "https://provider.example/v1"
    assert HTTPProvider(kind, base, "test-model", "test-key").generate("sources?", sources).claims


@pytest.mark.parametrize(
    "status,body",
    [
        (401, {"error": "secret-token"}),
        (429, {}),
        (500, {}),
        (200, {}),
        (200, {"choices": []}),
        (200, {"choices": [{"message": {"content": "not JSON"}}]}),
    ],
)
def test_provider_errors_are_redacted(monkeypatch, sources, status, body):
    mock_client(monkeypatch, lambda request: httpx.Response(status, json=body))
    with pytest.raises(ProviderError) as error:
        HTTPProvider("compatible", "https://provider.example/v1", "model", "secret-token").generate(
            "question", sources
        )
    assert "secret-token" not in str(error.value)


def test_provider_timeout(monkeypatch, sources):
    def handler(request):
        raise httpx.ReadTimeout("secret server details", request=request)

    mock_client(monkeypatch, handler)
    with pytest.raises(ProviderError):
        HTTPProvider("ollama", "http://127.0.0.1:11434", "model").generate("question", sources)


def test_provider_response_size_limit(monkeypatch, sources):
    mock_client(monkeypatch, lambda request: httpx.Response(200, content=b"x" * 256_001))
    with pytest.raises(ProviderError, match="size limit"):
        HTTPProvider("ollama", "http://127.0.0.1:11434", "model").generate("question", sources)


@pytest.mark.parametrize(
    "url",
    [
        "http://remote.example",
        "ftp://localhost",
        "https://key:secret@example.com",
        "https://example.com?token=secret",
        "",
    ],
)
def test_provider_url_validation(url):
    with pytest.raises(ValueError):
        HTTPProvider("compatible", url, "model")


def test_default_provider_needs_no_key(monkeypatch):
    monkeypatch.delenv("SOURCEBOOK_PROVIDER", raising=False)
    assert configured_provider().identity == "extractive-v1"
    monkeypatch.setenv("SOURCEBOOK_PROVIDER", "unknown")
    with pytest.raises(ValueError):
        configured_provider()


def test_split_spans_returns_verbatim_sentences_without_headings():
    text = "# Storage\n\nFirst sentence here. Second sentence here.\n\nAnother block sentence."
    spans = split_spans(text)
    assert spans == [
        "First sentence here.",
        "Second sentence here.",
        "Another block sentence.",
    ]
    assert all(span in text for span in spans)
    assert all(len(span) >= 8 for span in spans)


def test_split_spans_keeps_text_after_a_heading_with_no_blank_line():
    text = "## Storage\nThe upload limit is 20 MiB in total."
    assert split_spans(text) == ["The upload limit is 20 MiB in total."]


def test_split_spans_respects_the_quote_length_limit():
    text = ("word " * 500).strip()
    spans = split_spans(text)
    assert spans
    assert all(len(span) <= 1800 for span in spans)
    assert all(span in text for span in spans)


def span_sources():
    return [
        Source(
            id="s1",
            document_id="d1",
            name="guide.md",
            page=1,
            section="Storage",
            start=0,
            end=70,
            text="# Storage\n\nThe upload limit is 20 MiB in total.",
        )
    ]


@pytest.mark.parametrize("kind", ["ollama", "compatible"])
def test_span_provider_writes_the_quote_from_source_text(monkeypatch, kind):
    sources = span_sources()

    def handler(request):
        body = json.loads(request.content)
        assert body["messages"][0]["role"] == "system"
        assert "untrusted" in body["messages"][0]["content"]
        assert "sentence" in body["messages"][0]["content"]
        user = json.loads(body["messages"][1]["content"])
        assert user["evidence"][0] == {
            "source_id": "s1",
            "name": "guide.md",
            "sentences": [{"sentence": 0, "text": "The upload limit is 20 MiB in total."}],
        }
        content = json.dumps({"claims": [{"source_id": "s1", "sentence": 0}]})
        if kind == "ollama":
            assert request.url.path == "/api/chat"
            assert body["format"]["properties"]["claims"]["items"]["required"] == [
                "source_id",
                "sentence",
            ]
            return httpx.Response(200, json={"message": {"content": content}})
        assert request.url.path == "/v1/chat/completions"
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    mock_client(monkeypatch, handler)
    base = "http://localhost:11434" if kind == "ollama" else "https://provider.example/v1"
    draft = SpanProvider(kind, base, "test-model").generate("What is the upload limit?", sources)

    assert len(draft.claims) == 1
    claim = draft.claims[0]
    assert claim.text == claim.evidence[0].quote
    assert claim.evidence[0].quote in sources[0].text
    assert claim.evidence[0].source_id == "s1"


def test_span_provider_ignores_unknown_ids_out_of_range_and_duplicates(monkeypatch):
    sources = span_sources()
    content = json.dumps(
        {
            "claims": [
                {"source_id": "nope", "sentence": 0},
                {"source_id": "s1", "sentence": 9},
                {"source_id": "s1", "sentence": 0},
                {"source_id": "s1", "sentence": 0},
            ]
        }
    )
    mock_client(
        monkeypatch, lambda request: httpx.Response(200, json={"message": {"content": content}})
    )
    draft = SpanProvider("ollama", "http://127.0.0.1:11434", "model").generate("q", sources)
    assert len(draft.claims) == 1


def test_span_provider_abstains_without_calling_when_nothing_is_selectable(monkeypatch):
    sources = [
        Source(
            id="s1",
            document_id="d1",
            name="a.txt",
            page=1,
            section="",
            start=0,
            end=2,
            text="hi",
        )
    ]

    def handler(request):
        raise AssertionError("the provider must not be called when there is nothing to select")

    mock_client(monkeypatch, handler)
    draft = SpanProvider("ollama", "http://127.0.0.1:11434", "model").generate("q", sources)
    assert draft.claims == []


def test_span_provider_invalid_json_is_redacted(monkeypatch):
    sources = span_sources()
    mock_client(
        monkeypatch,
        lambda request: httpx.Response(200, json={"message": {"content": "not json"}}),
    )
    with pytest.raises(ProviderError) as error:
        SpanProvider("ollama", "http://127.0.0.1:11434", "model").generate("q", sources)
    assert "not json" not in str(error.value)


def test_provider_rejects_unknown_answer_style():
    with pytest.raises(ValueError):
        HTTPProvider("ollama", "http://127.0.0.1:11434", "model", style="nope")


def test_configured_provider_selects_the_span_style(monkeypatch):
    monkeypatch.setenv("SOURCEBOOK_PROVIDER", "ollama")
    monkeypatch.setenv("SOURCEBOOK_MODEL", "granite4-q8")
    monkeypatch.setenv("SOURCEBOOK_ANSWER_STYLE", "span")
    provider = configured_provider()
    assert isinstance(provider, SpanProvider)
    assert provider.identity.endswith(":span-v1")

    monkeypatch.setenv("SOURCEBOOK_ANSWER_STYLE", "quote")
    assert configured_provider().identity.endswith(":prompt-v1")

    monkeypatch.delenv("SOURCEBOOK_ANSWER_STYLE")
    assert configured_provider().identity.endswith(":prompt-v1")
