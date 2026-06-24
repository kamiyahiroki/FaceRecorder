# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-screen Android app (Java) that records a **face-registration video**: it walks the user through 5 head poses (front, right, left, up, down) using a guided wizard, detecting the head angle on-device with ML Kit Face Detection and recording a short clip per pose into one continuous `.mp4`. The output video is the enrollment input for a separate Python face-recognition pipeline (see below).

## Build & run

Uses the Gradle wrapper (`gradlew.bat` on Windows / `./gradlew` on POSIX). Single module: `:app`. No CI, lint config, or formatter beyond Gradle defaults.

```bash
gradlew.bat assembleDebug          # build debug APK
gradlew.bat installDebug           # build + install on connected device/emulator
gradlew.bat test                   # JVM unit tests (:app:testDebugUnitTest)
gradlew.bat connectedAndroidTest   # instrumented tests (needs a device/emulator)
gradlew.bat assembleRelease        # release APK (not minified; not signed by Gradle)
```

Run a single unit test class/method:
```bash
gradlew.bat test --tests "com.example.facerecorder.ExampleUnitTest.addition_isCorrect"
```

Notes:
- Only the template `ExampleUnitTest` / `ExampleInstrumentedTest` exist — there are no real tests yet.
- `minSdk 29`, `targetSdk 36`, Java 11. Requires the camera permission at runtime; the app finishes if denied.
- The app is locked to `portrait` and keeps the screen on while open.
- A keystore exists at `keystore/facerecorder.jks`, but `app/build.gradle.kts` defines **no `signingConfig`** — release signing is not wired into the build. Dependency versions are centralized in `gradle/libs.versions.toml`.

## Architecture

Three source files do all the work; everything else is generated Android scaffolding.

- [MainActivity.java](app/src/main/java/com/example/facerecorder/MainActivity.java) — the entire wizard: CameraX setup, per-frame face-angle analysis, pose state machine, recording, and audio prompts.
- [FaceGuideView.java](app/src/main/java/com/example/facerecorder/FaceGuideView.java) — a custom `View` that darkens everything outside a head-shaped oval as a framing guide.
- [Mp4Encoder.java](app/src/main/java/com/example/facerecorder/Mp4Encoder.java) — a self-managed `MediaCodec` (H.264) + `MediaMuxer` wrapper that encodes upright `Bitmap` frames into one `.mp4` at a guaranteed exact resolution.

### The wizard flow (one encoder, five poses)

The UI is a `FrameLayout` ([activity_main.xml](app/src/main/res/layout/activity_main.xml)) with three stacked panels toggled by visibility: `panelInput` (name entry) → `panelRecording` (overlay during capture) → `panelComplete` (save path). A single `Mp4Encoder` runs continuously across all five poses; `nextPose()` advances the wizard and only calls `encoder.finish()` after the last pose, producing one `<name>_<timestamp>.mp4` in `getExternalFilesDir(MOVIES)`.

### Recording = self-encoded frames (not CameraX VideoCapture)

CameraX `VideoCapture<Recorder>` can only request logical `Quality` presets (SD/HD/…) whose pixel size is device-dependent, so it can't guarantee an exact resolution. Instead the app encodes frames itself:

- `bindCamera()` binds two CameraX use cases via a `UseCaseGroup`: `Preview` and `ImageAnalysis` (keep-only-latest, **1280×720** `ANALYSIS_SIZE` via `ResolutionSelector`). It prefers the front camera, falling back to back. There is no `VideoCapture` use case.
- The **same** `ImageAnalysis` frames feed both ML Kit detection and the encoder.
- **What gets recorded — mirrors recorder.py exactly.** Frames are encoded in *both* phases: (1) during `WAITING` (pose adjustment), **every frame in which a face is detected** is encoded — this matches `recorder.py:621`, which writes every face-detected frame so register.py video-mode sees the full range of in-between head angles, not just the five target poses; (2) during `HOLDING`, the fixed `FRAMES_PER_RECORDING` (45) frames are encoded with no pose re-check. The only intentional divergence: a too-small face (`MIN_FACE_FRACTION`) is **not** recorded and blocks pose progression — recorder.py's Windows detectors don't gate on size, but this matches the Raspberry Pi/Hailo target (`min_face_size=64`).
- **Portrait hold, 640×480 landscape output — reconciled by a shared `ViewPort`.** The phone is portrait-locked, but the downstream pipeline wants a 640×480 landscape, upright-face clip. So the camera captures large (1280×720) and a `ViewPort` derived from the on-screen **4:3 camera band** (`cameraBand`, resized in code to a centered `width × width*3/4` strip) crops *both* preview and analysis to the same window. `previewView` uses `fillCenter`, so **what the user sees is exactly what gets recorded.**
- Per recorded frame (`imageProxyToUprightBitmap`): YUV_420_888 → NV21 → JPEG → `Bitmap`, then **cropped to `ImageProxy.getCropRect()`** (the ViewPort window), rotated upright by `rotationDegrees` (front portrait ≈ 270°), and finally scaled by the encoder to **exactly 640×480**. Mirroring is **off** (matches the old `MIRROR_MODE_OFF`). The JPEG round-trip is a deliberate simplicity/robustness trade-off; it caps effective recording FPS.
- Camera start is deferred through two nested `cameraBand.post(...)` calls so the band is resized **and** its layout pass has completed before `bindCamera()` reads `previewView.getViewPort()` — otherwise the crop would be derived from the full-screen (portrait) size. Preserve this ordering.

### Threading model

- `analyzeFrame()` runs on a single-thread `cameraExecutor`; UI updates are marshalled with `runOnUiThread`.
- **Detection is synchronous.** During `WAITING`, `analyzeFrame()` calls `Tasks.await(faceDetector.process(...))` on the `cameraExecutor` thread (not the async listener), so detection → frame encoding → the `WAITING → HOLDING` transition all run on one thread. `onFacesDetected()` therefore runs on `cameraExecutor` (not the main thread) — which is why `currentPose` is `volatile`.
- **All `Mp4Encoder` access stays on `cameraExecutor`**: `encode()` (via `encodeCurrentFrame()`) is called only from `analyzeFrame()`/`onFacesDetected()`, and `finish()` is posted to the *same* executor from `nextPose()`/`onDestroy()` so the single-thread queue guarantees it runs after the last queued frame. The encoder is constructed on the UI thread in `startRecording()` (setup only; no concurrent use).
- Cross-thread state (`poseState`, `recordingFramesLeft`, `holdFrameCount`, `isRecording`, `currentPose`) is `volatile`. **The write ordering in the `WAITING → HOLDING` transition is load-bearing and documented in code**: `recordingFramesLeft` must be set *before* the `volatile` write to `poseState`, or the camera thread can decrement a stale `0` and skip the pose without recording. Preserve that ordering when editing.
- During `HOLDING`, face analysis is skipped; each frame is encoded then counted down (`countRecordingFrame()`) — mirroring recorder.py's `_STATE_RECORDING`, which records a fixed frame count with no pose re-check.

### Pose detection is a deliberate port — keep it in sync

This is the most important thing to understand before touching detection logic. The angle math and all thresholds in `MainActivity.java` are a **direct port of a separate Python `face_recognition` project** (`register.py`, `recorder.py`, `face_processor.py` — not in this repo). The code comments repeatedly cite those files as the source of truth:

- `estimateYaw()` / `estimatePitch()` replicate `FaceProcessor.estimate_yaw_ratio()` / `estimate_pitch_ratio()`, including the `0.495` frontal-pitch baseline and the use of the **NOSE_BRIDGE contour's last point** as the nose tip (chosen to match SCRFD's 5-point landmark output).
- Constants mirror named values in the Python code: `FRAMES_PER_RECORDING=45`, `HOLD_FRAMES=3`, `MIN_FACE_FRACTION` (≈ `min_face_size=64`), and the yaw/pitch thresholds.
- `UPDOWN_PITCH_MIN = 0.17f` intentionally **diverges** from the Python value (`0.09`) to compensate for a measured offset between ML Kit's landmarks and SCRFD's. The long comment at `MainActivity.java:69-88` explains why and flags it as needing on-device tuning.

When changing detection behavior, treat consistency with that Python pipeline as a hard constraint — the recorded video must satisfy the *Python* acceptance checks downstream, not just look right on the phone. ML Kit requires `PERFORMANCE_MODE_ACCURATE` to expose the contour/landmark data this relies on.

### Audio prompts

Each pose plays a matching MP3 from `res/raw/` (`face_forward`, `turn_right`, `turn_left`, `tilt_up`, `tilt_down`) via a single recycled `MediaPlayer` in `playPoseSound()`. Pose text, pose keys, and raw-resource IDs are three parallel arrays (`POSES`, `POSE_KEYS`, `POSE_RAW_IDS`) indexed by `currentPose` — keep them aligned and same-length when adding or reordering poses.

## Conventions

- UI strings and code comments are in Japanese; comments cross-referencing the Python pipeline are in English. Match the surrounding style.
- The package is `com.example.facerecorder` (the AGP template default); it has not been renamed.
