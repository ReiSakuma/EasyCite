from __future__ import annotations

from citation_agent.zotero import ZoteroClient


def _item(index: int) -> dict:
    return {
        "key": f"ITEM{index}",
        "data": {
            "key": f"ITEM{index}",
            "itemType": "journalArticle",
            "title": f"Paper {index}",
            "abstractNote": f"Abstract evidence for paper {index}.",
        },
    }


def test_list_scope_paginates_selected_collection(monkeypatch) -> None:
    client = ZoteroClient(
        "http://127.0.0.1:23119/api",
        collection_keys=["COLLECTION_A"],
    )
    requested_urls: list[str] = []

    def request(url: str):
        requested_urls.append(url)
        if "start=0" in url:
            return [_item(index) for index in range(100)]
        if "start=100" in url:
            return [_item(index) for index in range(100, 105)]
        return []

    monkeypatch.setattr(client, "_request_url_json", request)
    monkeypatch.setattr(client, "_fetch_better_bibtex_citation_keys", lambda _items: {})
    monkeypatch.setattr(client, "_resolve_pdf_path", lambda _item_value: None)

    suggestions = client.list_scope(limit=150)

    assert len(suggestions) == 105
    assert any("collections/COLLECTION_A/items/top" in url for url in requested_urls)
    assert any("start=100" in url for url in requested_urls)


def test_list_scope_can_defer_pdf_attachment_requests(monkeypatch) -> None:
    client = ZoteroClient("http://127.0.0.1:23119/api")
    monkeypatch.setattr(client, "_request_url_json", lambda _url: [_item(1)])
    monkeypatch.setattr(client, "_fetch_better_bibtex_citation_keys", lambda _items: {})
    calls: list[str] = []
    monkeypatch.setattr(
        client,
        "_resolve_pdf_path",
        lambda item: calls.append(str(item["key"])) or "/tmp/paper.pdf",
    )

    suggestions = client.list_scope(limit=1, resolve_pdf_paths=False)
    paths = client.resolve_pdf_paths([suggestions[0].paper_id])

    assert calls == ["ITEM1"]
    assert paths == {suggestions[0].paper_id: "/tmp/paper.pdf"}
