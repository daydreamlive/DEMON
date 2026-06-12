import json

import pytest

from demos.common import static_site


def _write_demo(
    tmp_path,
    route="/demo",
    manifest_name="demon.demo.json",
    dirname=None,
    **extra,
):
    demo = tmp_path / (dirname or route.strip("/"))
    demo.mkdir(parents=True)
    (demo / "index.html").write_text("index", encoding="utf-8")
    manifest = {"route": route, **extra}
    (demo / manifest_name).write_text(json.dumps(manifest), encoding="utf-8")
    return demo


def test_loads_directory_manifest(tmp_path):
    demo = _write_demo(tmp_path, route="/external")

    mount = static_site.load_static_demo(demo)

    assert mount.route == "/external"
    assert mount.root == demo.resolve()
    assert mount.entry == "index.html"


def test_loads_manifest_path_with_legacy_name(tmp_path):
    demo = _write_demo(tmp_path, route="/legacy", manifest_name="demo.static.json")

    mount = static_site.load_static_demo(demo / "demo.static.json")

    assert mount.route == "/legacy"
    assert mount.root == demo.resolve()


def test_defaults_entry_to_index_html(tmp_path):
    demo = _write_demo(tmp_path, route="/default")

    assert static_site.load_static_demo(demo).entry == "index.html"


def test_serves_route_redirect_and_route_entry(tmp_path):
    demo = _write_demo(tmp_path, route="/demo", entry="home.html")
    (demo / "home.html").write_text("home", encoding="utf-8")
    mount = static_site.load_static_demo(demo)
    mounts = {mount.route: mount}

    redirect = static_site.serve_static_mounts("/demo", mounts)
    entry = static_site.serve_static_mounts("/demo/", mounts)

    assert redirect is not None
    assert redirect.status_code == 301
    assert redirect.headers["Location"] == "/demo/"
    assert entry is not None
    assert entry.status_code == 200
    assert entry.body == b"home"


def test_blocks_traversal(tmp_path):
    demo = _write_demo(tmp_path, route="/demo")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    mount = static_site.load_static_demo(demo)

    assert static_site.serve_static_mounts("/demo/../secret.txt", {"/demo": mount}) is None

    (demo / "demon.demo.json").write_text(
        json.dumps({"route": "/bad", "entry": "../secret.txt"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes"):
        static_site.load_static_demo(demo)


@pytest.mark.parametrize("route", ["/api", "/api/demo", "/fixtures", "/sdk"])
def test_rejects_reserved_routes(tmp_path, route):
    demo = _write_demo(tmp_path, route=route)

    with pytest.raises(ValueError, match="reserved routes"):
        static_site.load_static_demo(demo)


@pytest.mark.parametrize("route", ["/", "///", "demo"])
def test_rejects_invalid_routes(tmp_path, route):
    demo = _write_demo(tmp_path, route=route, dirname="invalid")

    with pytest.raises(ValueError, match="non-root"):
        static_site.load_static_demo(demo)


def test_rejects_duplicate_routes(tmp_path, monkeypatch):
    first = _write_demo(tmp_path, route="/dupe", dirname="first")
    second = _write_demo(tmp_path, route="/dupe", dirname="second")
    second_manifest = second / "demon.demo.json"

    monkeypatch.setattr(static_site, "discover_static_demos", lambda: {})

    with pytest.raises(ValueError, match="already claimed"):
        static_site.build_static_mounts([first, second_manifest])
