"""Only a dedicated cloud HTTP graph service may freeze its startup heap."""

import gc
import sys
from types import SimpleNamespace

import pytest

from yamkit.inference import modal_service


@pytest.fixture
def fake_gc(monkeypatch):
    calls = []

    class FakeGC:
        enabled = True
        thresholds = (700, 10, 10)
        count = 123456
        failure = None

        def isenabled(self):
            return self.enabled

        def get_threshold(self):
            return self.thresholds

        def collect(self, generation):
            calls.append(("collect", generation))
            if self.failure == "collect":
                raise RuntimeError("fake collection failure")
            return 42

        def freeze(self):
            calls.append(("freeze",))
            if self.failure == "freeze":
                raise RuntimeError("fake freeze failure")
            if self.failure == "disabled":
                self.enabled = False
            if self.failure == "thresholds":
                self.thresholds = (1000, 10, 10)

        def get_freeze_count(self):
            return self.count

    value = FakeGC()
    monkeypatch.setattr(modal_service, "gc", value)
    monkeypatch.setenv("YAMKIT_ROOT", "/opt/yamkit")
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(is_local=lambda: False))
    return value, calls


def runtime():
    return SimpleNamespace(profile=SimpleNamespace(id="molmoact2"), execution_mode="cuda_graph10",
                           device="cuda", _prediction_count=0)


def test_preparation_collects_then_freezes_without_changing_automatic_gc(fake_gc):
    fake, calls = fake_gc
    actual_gc_enabled, actual_frozen_count = gc.isenabled(), gc.get_freeze_count()
    metadata = modal_service._prepare_http_graph_gc(runtime())
    assert calls == [("collect", 2), ("freeze",)]
    assert fake.enabled and fake.thresholds == (700, 10, 10)
    assert metadata["strategy"] == "freeze_after_model_load_v1"
    assert metadata["automatic_gc_enabled"] and metadata["frozen_objects"] == 123456
    assert metadata["collected_objects"] == 42
    assert metadata["collection_s"] >= 0 and metadata["freeze_s"] >= 0
    assert gc.isenabled() is actual_gc_enabled and gc.get_freeze_count() == actual_frozen_count


@pytest.mark.parametrize("context", ["local", "unknown_modal", "wrong_root", "cpu", "eager", "wrong_model", "already_predicted"])
def test_local_or_nonstartup_context_never_collects_or_freezes(fake_gc, monkeypatch, context):
    _, calls = fake_gc
    value = runtime()
    if context == "local":
        monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(is_local=lambda: True))
    elif context == "unknown_modal":
        monkeypatch.setitem(sys.modules, "modal", SimpleNamespace())
    elif context == "wrong_root":
        monkeypatch.setenv("YAMKIT_ROOT", "/home/andre/rohan-new")
    elif context == "cpu":
        value.device = "cpu"
    elif context == "eager":
        value.execution_mode = "eager"
    elif context == "wrong_model":
        value.profile.id = "smolvla"
    else:
        value._prediction_count = 1
    with pytest.raises(RuntimeError, match="dedicated Modal"):
        modal_service._prepare_http_graph_gc(value)
    assert calls == []


@pytest.mark.parametrize("failure", ["collect", "freeze", "disabled", "thresholds", "empty_frozen", "initially_disabled"])
def test_gc_preparation_failure_cannot_produce_readiness(fake_gc, failure):
    fake, calls = fake_gc
    fake.failure = failure
    if failure == "empty_frozen":
        fake.count = 0
    if failure == "initially_disabled":
        fake.enabled = False
    with pytest.raises(RuntimeError):
        modal_service._prepare_http_graph_gc(runtime())
    if failure == "initially_disabled":
        assert calls == []
    elif failure == "collect":
        assert calls == [("collect", 2)]


@pytest.fixture
def service_factory(monkeypatch, fake_gc):
    events = fake_gc[1]
    captured = {}

    class Image:
        @classmethod
        def debian_slim(cls, **kwargs):
            return cls()

        def pip_install_from_requirements(self, *args):
            return self

        def env(self, *args):
            return self

        def add_local_dir(self, *args, **kwargs):
            return self

        def add_local_file(self, *args):
            return self

    class App:
        def __init__(self, name):
            self.name = name

        def cls(self, **kwargs):
            def decorate(value):
                captured["class"] = value
                return value
            return decorate

    fake = SimpleNamespace(
        is_local=lambda: False, App=App, Image=Image,
        Volume=SimpleNamespace(from_name=lambda *a, **k: SimpleNamespace(commit=lambda: events.append(("commit",)))),
        Secret=SimpleNamespace(from_dict=lambda values: values), enter=lambda: lambda f: f,
        method=lambda: lambda f: f, asgi_app=lambda: lambda f: f)
    monkeypatch.setitem(sys.modules, "modal", fake)

    def load(profile, *, device, execution_mode):
        events.append(("load", profile, execution_mode))
        value = runtime()
        value.execution_mode = execution_mode
        value.ready = lambda: {"ready": True}
        value.reset = lambda session_id: events.append(("reset", session_id))
        return value

    def build_id():
        events.append(("identity",))
        return "test-build"

    monkeypatch.setattr("yamkit.inference.service.ModelRuntime.load", load)
    monkeypatch.setattr("yamkit.inference.identity.inference_build_id", build_id)

    def create(*, transport="http", mode="cuda_graph10"):
        modal_service.create_app("molmoact2", transport=transport, execution_mode=mode,
                                 http_token="t" * 48 if transport == "http" else None)
        return captured["class"]()

    return create, events


def test_factory_freezes_only_after_complete_load_before_readiness_and_never_on_reset(service_factory):
    create, events = service_factory
    service = create()
    assert not events, "Defining a Modal app must not alter local GC"
    service.load()
    assert events == [("load", "molmoact2", "cuda_graph10"), ("commit",), ("identity",), ("collect", 2), ("freeze",)]
    assert service.ready()["python_gc"]["strategy"] == "freeze_after_model_load_v1"
    service.reset("finished-session")
    assert events[-1] == ("reset", "finished-session") and events.count(("freeze",)) == 1


@pytest.mark.parametrize("transport,mode", [("sdk", "cuda_graph10"), ("sdk", "eager"), ("http", "eager")])
def test_other_services_keep_existing_gc_behavior(service_factory, transport, mode):
    create, events = service_factory
    service = create(transport=transport, mode=mode)
    service.load()
    assert all(event[0] not in ("collect", "freeze") for event in events)
    assert service.ready()["python_gc"] == {"strategy": "automatic"}


def test_freeze_error_propagates_from_container_startup(service_factory, fake_gc):
    create, events = service_factory
    fake_gc[0].failure = "freeze"
    service = create()
    with pytest.raises(RuntimeError, match="fake freeze"):
        service.load()
    assert events[-1] == ("freeze",)
    assert not hasattr(service, "container_init_s")
