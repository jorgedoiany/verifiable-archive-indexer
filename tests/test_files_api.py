import httpx
import pytest

from app.config import Settings
from app.pipeline import files_api
from app.pipeline.files_api import FilesError, delete_file, upload_video

FILES_URL = "https://api.meta.ai/v1/files"
UPLOADED = {"id": "file-842549258569145", "object": "file", "status": "uploaded"}


@pytest.fixture
def settings():
    return Settings(
        api_key="test-key",
        api_base="https://api.meta.ai/v1",
        spark_model="muse-spark-1.3",
        wonder_model="muse-wonder-1.0",
        asr_model="muse-voice-transcribe-1.0",
        sam_model="sam-3.1",
    )


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not-a-real-video")
    return path


def test_upload_sends_documented_form(clip, settings, respx_mock):
    route = respx_mock.post(FILES_URL).mock(
        return_value=httpx.Response(200, json=UPLOADED)
    )
    file_id = upload_video(clip, settings, expires_seconds=7200)
    assert file_id == "file-842549258569145"

    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer test-key"
    body = request.content
    assert b'name="purpose"' in body and b"user_data" in body
    assert b'name="expires_after[anchor]"' in body and b"created_at" in body
    assert b'name="expires_after[seconds]"' in body and b"7200" in body
    assert b'filename="clip.mp4"' in body
    assert body.index(b'name="purpose"') < body.index(b'name="file"')
    assert body.index(b'name="expires_after[seconds]"') < body.index(b'name="file"')


def test_rejects_non_mp4(tmp_path, settings):
    other = tmp_path / "clip.mov"
    other.write_bytes(b"x")
    with pytest.raises(ValueError):
        upload_video(other, settings)


def test_rejects_oversized_file(clip, settings, monkeypatch):
    monkeypatch.setattr(files_api, "MAX_UPLOAD_BYTES", 4)
    with pytest.raises(ValueError):
        upload_video(clip, settings)


@pytest.mark.parametrize("seconds", [3_599, 2_592_001])
def test_rejects_expiry_out_of_range(clip, settings, seconds):
    with pytest.raises(ValueError):
        upload_video(clip, settings, expires_seconds=seconds)


def test_retries_on_429_then_succeeds(clip, settings, respx_mock):
    route = respx_mock.post(FILES_URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json=UPLOADED)]
    )
    waits = []
    assert upload_video(clip, settings, sleep=waits.append).startswith("file-")
    assert route.call_count == 2
    assert waits == [1]


def test_client_error_is_not_retried(clip, settings, respx_mock):
    route = respx_mock.post(FILES_URL).mock(
        return_value=httpx.Response(400, text="storage limit")
    )
    with pytest.raises(FilesError) as info:
        upload_video(clip, settings, sleep=lambda s: None)
    assert info.value.status_code == 400
    assert route.call_count == 1


def test_response_without_file_id_is_rejected(clip, settings, respx_mock):
    respx_mock.post(FILES_URL).mock(return_value=httpx.Response(200, json={"x": 1}))
    with pytest.raises(FilesError):
        upload_video(clip, settings)


def test_delete_sends_request(settings, respx_mock):
    route = respx_mock.delete(f"{FILES_URL}/file-1").mock(
        return_value=httpx.Response(200, json={"deleted": True})
    )
    delete_file("file-1", settings)
    assert route.calls.last.request.headers["Authorization"] == "Bearer test-key"


def test_delete_missing_file_is_ignored_by_default(settings, respx_mock):
    respx_mock.delete(f"{FILES_URL}/file-1").mock(return_value=httpx.Response(404))
    delete_file("file-1", settings)
    with pytest.raises(FilesError):
        delete_file("file-1", settings, ignore_missing=False)


def test_delete_server_error_raises(settings, respx_mock):
    respx_mock.delete(f"{FILES_URL}/file-1").mock(
        return_value=httpx.Response(500, text="boom")
    )
    with pytest.raises(FilesError):
        delete_file("file-1", settings)
