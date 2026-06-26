package com.example.facerecorder;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.content.res.ColorStateList;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.PointF;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.Environment;
import android.util.Size;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.UseCaseGroup;
import androidx.camera.core.ViewPort;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.google.android.gms.tasks.Tasks;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceContour;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.google.mlkit.vision.face.FaceLandmark;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    // Pose sequence — mirrors POSE_SEQUENCE in face_recognition/src/register.py
    private static final String[] POSES = {
        "正面を向いてください",
        "ゆっくり右を向いてください",
        "ゆっくり左を向いてください",
        "ゆっくり上を向いてください",
        "ゆっくり下を向いてください"
    };
    private static final String[] POSE_KEYS = {
        "front", "right", "left", "up", "down"
    };

    // Normalized thresholds for pose detection.
    // yaw  = (nose_x - eye_mid_x) / inter_ocular  (negative = user's own right)
    // pitch = (nose_y - eye_mid_y) / eye_to_mouth - 0.495  (negative = looking up)
    //
    // register.py / recorder.py use UPDOWN_PITCH_MIN = 0.09 for final acceptance.
    // ML Kit's NOSE_BRIDGE endpoint and MOUTH landmarks sit at slightly different
    // positions from SCRFD's 5-point output, causing ML Kit pitch to be
    // consistently higher (more positive) than SCRFD for the same physical tilt.
    // Empirically this makes FaceRecorder accept "down" too early — the user
    // reaches ML Kit pitch=0.09 while SCRFD pitch is still lower.
    // Setting UPDOWN_PITCH_MIN to 0.17 compensates so that when ML Kit says OK the
    // user has tilted far enough that SCRFD also clears its 0.09 threshold.
    // NOTE: 0.17 targets a SCRFD-equivalent of ~0.12, which is stricter than the
    // 0.09 register.py actually requires — verify on-device whether 0.17 is right
    // or whether ~0.14 (0.09 + ML Kit offset) raises up/down success without
    // letting register.py reject the pose.
    private static final float FRONT_YAW_MAX    = 0.12f;
    private static final float SIDE_YAW_MIN     = 0.18f;
    private static final float FRONT_PITCH_MAX  = 0.06f;
    private static final float UPDOWN_PITCH_MIN = 0.17f;  // register.py uses 0.09; extra margin for ML Kit offset

    // Minimum face bounding-box width as a fraction of the analysis frame width.
    // Mirrors min_face_size=64 in face_processor.py — ensures ArcFace receives
    // a face large enough to extract a quality embedding.
    // 20% of a typical 640 px analysis frame ≈ 128 px minimum face width.
    private static final float MIN_FACE_FRACTION = 0.20f;

    // Frames to record per pose after hold confirmation.
    // Mirrors FRAMES_PER_RECORDING=45 in recorder.py — frame-count-based so the
    // captured frame count is device-independent (time-based varies with CPU load).
    private static final int FRAMES_PER_RECORDING = 45;

    // Consecutive frames at target pose required before recording starts.
    // Mirrors HOLD_FRAMES=3 in recorder.py — filters single-frame pose misdetections.
    private static final int HOLD_FRAMES = 3;

    // Output video geometry. The phone is held portrait, but the desired output is a
    // 640×480 landscape, upright-face clip (the format the downstream Python pipeline
    // expects). To reconcile the two, the camera captures a larger frame, a shared
    // CameraX ViewPort crops both preview and analysis to a 4:3 landscape window, and
    // each recorded frame is cropped to that window, rotated upright, and scaled to
    // exactly 640×480. "Capture large, crop the center 4:3" — preview == recording.
    private static final int OUT_WIDTH  = 640;
    private static final int OUT_HEIGHT = 480;
    private static final int ENC_FPS     = 30;
    private static final int ENC_BITRATE = 4_000_000;

    // Analysis capture resolution (sensor-natural landscape). Larger than 640×480 so
    // the 4:3 crop still has enough pixels to downscale cleanly to the output size.
    private static final Size ANALYSIS_SIZE = new Size(1280, 720);

    private static final int[] POSE_RAW_IDS = {
        R.raw.face_forward,
        R.raw.turn_right,
        R.raw.turn_left,
        R.raw.tilt_up,
        R.raw.tilt_down
    };

    private enum PoseState { WAITING, HOLDING, TRANSITIONING }

    // Views
    private PreviewView previewView;
    private FrameLayout cameraBand;
    private View panelInput, panelRecording, panelComplete;
    private EditText etName;
    private TextView tvPoseProgress, tvInstruction, tvCountdown, tvAngleStatus, tvFilePath;
    private ProgressBar progressBar;

    // Audio
    private MediaPlayer mediaPlayer;

    // CameraX + self-managed encoder
    private FaceDetector faceDetector;
    private ExecutorService cameraExecutor;
    private Mp4Encoder encoder;
    private String outputFilePath;

    // Wizard state
    // currentPose is volatile because detection now runs on cameraExecutor (see
    // analyzeFrame): it is written on the main thread (nextPose/updatePoseUI) and read
    // on the camera thread.
    private volatile int currentPose = 0;
    private volatile PoseState poseState = PoseState.WAITING;
    private volatile int recordingFramesLeft = 0;
    private volatile int holdFrameCount = 0;
    private volatile boolean isRecording = false;

    private final ActivityResultLauncher<String> permissionLauncher =
        registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> {
            if (granted) {
                startCamera();
            } else {
                Toast.makeText(this, "カメラの許可が必要です", Toast.LENGTH_LONG).show();
                finish();
            }
        });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        previewView    = findViewById(R.id.previewView);
        cameraBand     = findViewById(R.id.cameraBand);
        panelInput     = findViewById(R.id.panelInput);
        panelRecording = findViewById(R.id.panelRecording);
        panelComplete  = findViewById(R.id.panelComplete);
        etName         = findViewById(R.id.etName);
        tvPoseProgress = findViewById(R.id.tvPoseProgress);
        tvInstruction  = findViewById(R.id.tvInstruction);
        tvCountdown    = findViewById(R.id.tvCountdown);
        tvAngleStatus  = findViewById(R.id.tvAngleStatus);
        tvFilePath     = findViewById(R.id.tvFilePath);
        progressBar    = findViewById(R.id.progressBar);

        Button btnStart = findViewById(R.id.btnStart);
        Button btnClose = findViewById(R.id.btnClose);
        btnStart.setOnClickListener(v -> onStartClicked());
        btnClose.setOnClickListener(v -> finish());

        cameraExecutor = Executors.newSingleThreadExecutor();

        // CONTOUR_MODE_ALL: needed to get nose tip from NOSE_BRIDGE contour.
        // LANDMARK_MODE_ALL: needed for eye and mouth landmark positions.
        // Both require PERFORMANCE_MODE_ACCURATE (ML Kit restriction).
        FaceDetectorOptions options = new FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .setMinFaceSize(0.15f)
            .enableTracking()
            .build();
        faceDetector = FaceDetection.getClient(options);

        // Resize the camera band to a centered 4:3 landscape strip once the layout is
        // measured, then (after the resize's layout pass) start the camera. Binding must
        // happen after the resize so the ViewPort derived from the band — and thus the
        // analysis crop used for recording — has its final 4:3 dimensions.
        cameraBand.post(() -> {
            resizeCameraBandTo4x3();
            cameraBand.post(() -> {
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                        == PackageManager.PERMISSION_GRANTED) {
                    startCamera();
                } else {
                    permissionLauncher.launch(Manifest.permission.CAMERA);
                }
            });
        });
    }

    // ── Camera setup ──────────────────────────────────────────────────────────

    // Shrink the camera band to a horizontal 4:3 strip (width × width*3/4) centered on
    // the portrait screen. The preview and face guide live inside it, so what the user
    // sees is exactly the 640×480 region that gets recorded.
    private void resizeCameraBandTo4x3() {
        int w = cameraBand.getWidth();
        if (w <= 0) return;
        FrameLayout.LayoutParams lp = (FrameLayout.LayoutParams) cameraBand.getLayoutParams();
        lp.width = w;
        lp.height = w * OUT_HEIGHT / OUT_WIDTH;  // 4:3 (480/640)
        lp.gravity = Gravity.CENTER;
        cameraBand.setLayoutParams(lp);
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> future =
            ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            try {
                bindCamera(future.get());
            } catch (ExecutionException | InterruptedException e) {
                Toast.makeText(this, "カメラの初期化に失敗しました", Toast.LENGTH_LONG).show();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindCamera(ProcessCameraProvider provider) {
        provider.unbindAll();

        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        // Capture larger than the output, then crop to the band's 4:3 window.
        ResolutionSelector resolutionSelector = new ResolutionSelector.Builder()
            .setResolutionStrategy(new ResolutionStrategy(
                ANALYSIS_SIZE,
                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
            .build();

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
            .setResolutionSelector(resolutionSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build();
        imageAnalysis.setAnalyzer(cameraExecutor, this::analyzeFrame);

        CameraSelector selector;
        try {
            selector = provider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA)
                ? CameraSelector.DEFAULT_FRONT_CAMERA
                : CameraSelector.DEFAULT_BACK_CAMERA;
        } catch (androidx.camera.core.CameraInfoUnavailableException e) {
            selector = CameraSelector.DEFAULT_BACK_CAMERA;
        }

        // A shared ViewPort derived from the 4:3 band crops BOTH preview and analysis to
        // the same window, so ImageProxy.getCropRect() on each analysis frame is exactly
        // the region shown in the preview. The encoder uses that crop → preview == output.
        ViewPort viewPort = previewView.getViewPort();
        UseCaseGroup.Builder groupBuilder = new UseCaseGroup.Builder()
            .addUseCase(preview)
            .addUseCase(imageAnalysis);
        if (viewPort != null) groupBuilder.setViewPort(viewPort);

        provider.bindToLifecycle(this, selector, groupBuilder.build());
    }

    @SuppressLint("UnsafeOptInUsageError")
    private void analyzeFrame(ImageProxy imageProxy) {
        if (!isRecording) {
            imageProxy.close();
            return;
        }
        // During the confirmed recording phase, skip face detection: encode the frame
        // and count it down. Mirrors _STATE_RECORDING in recorder.py which writes frames
        // without any pose check — the recording runs to FRAMES_PER_RECORDING regardless.
        if (poseState == PoseState.HOLDING) {
            try {
                encodeCurrentFrame(imageProxy);
            } finally {
                imageProxy.close();
            }
            countRecordingFrame();
            return;
        }
        if (poseState == PoseState.TRANSITIONING || imageProxy.getImage() == null) {
            imageProxy.close();
            return;
        }
        // WAITING (pose-adjustment phase). Run detection SYNCHRONOUSLY on this same
        // cameraExecutor thread so detection, frame encoding, and the WAITING→HOLDING
        // transition all happen on one thread — keeping all Mp4Encoder access single-
        // threaded. Mirrors recorder.py's _STATE_POSE, which detects and writes frames
        // on its single update loop.
        int rot = imageProxy.getImageInfo().getRotationDegrees();
        Rect crop = imageProxy.getCropRect();
        int frameW = (rot == 90 || rot == 270) ? crop.height() : crop.width();
        InputImage image = InputImage.fromMediaImage(imageProxy.getImage(), rot);
        List<Face> faces;
        try {
            faces = Tasks.await(faceDetector.process(image));
        } catch (Exception e) {
            imageProxy.close();
            return;
        }
        try {
            onFacesDetected(faces, frameW, imageProxy);
        } finally {
            imageProxy.close();
        }
    }

    // ── Frame conversion for encoding ─────────────────────────────────────────
    // Converts a YUV_420_888 analysis frame into an upright ARGB bitmap sized for
    // the encoder. The caller is responsible for closing the ImageProxy.

    private Bitmap imageProxyToUprightBitmap(ImageProxy image) {
        int w = image.getWidth();
        int h = image.getHeight();
        int rotation = image.getImageInfo().getRotationDegrees();
        Rect crop = image.getCropRect();

        byte[] nv21 = yuv420ToNv21(image);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, w, h, null);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, w, h), 95, baos);
        byte[] jpeg = baos.toByteArray();
        Bitmap decoded = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.length);
        if (decoded == null) return null;

        // Crop to the shared ViewPort window (the region shown in the preview). cropRect
        // is in the unrotated buffer's coordinates, so crop first, then rotate upright.
        Bitmap cropped;
        if (crop != null && (crop.width() < w || crop.height() < h)) {
            cropped = Bitmap.createBitmap(decoded, crop.left, crop.top,
                crop.width(), crop.height());
            decoded.recycle();
        } else {
            cropped = decoded;
        }

        // Rotate to upright. Front-camera portrait typically reports 270°, turning the
        // landscape crop into an upright frame. Mirroring is intentionally OFF (matches
        // the previous VideoCapture MIRROR_MODE_OFF behavior). The encoder then scales
        // the result to exactly OUT_WIDTH×OUT_HEIGHT (640×480).
        Bitmap upright;
        if (rotation != 0) {
            Matrix matrix = new Matrix();
            matrix.postRotate(rotation);
            upright = Bitmap.createBitmap(cropped, 0, 0,
                cropped.getWidth(), cropped.getHeight(), matrix, true);
            cropped.recycle();
        } else {
            upright = cropped;
        }
        return upright;
    }

    // Converts an ImageProxy in YUV_420_888 to a contiguous NV21 byte array,
    // honoring per-plane row/pixel strides. Absolute buffer indexing keeps it
    // correct regardless of buffer position.
    private static byte[] yuv420ToNv21(ImageProxy image) {
        int w = image.getWidth();
        int h = image.getHeight();
        byte[] nv21 = new byte[w * h * 3 / 2];

        ImageProxy.PlaneProxy yPlane = image.getPlanes()[0];
        ImageProxy.PlaneProxy uPlane = image.getPlanes()[1];
        ImageProxy.PlaneProxy vPlane = image.getPlanes()[2];

        ByteBuffer yBuf = yPlane.getBuffer();
        int yRowStride = yPlane.getRowStride();
        int yPixStride = yPlane.getPixelStride();
        int pos = 0;
        for (int row = 0; row < h; row++) {
            int base = row * yRowStride;
            for (int col = 0; col < w; col++) {
                nv21[pos++] = yBuf.get(base + col * yPixStride);
            }
        }

        ByteBuffer uBuf = uPlane.getBuffer();
        ByteBuffer vBuf = vPlane.getBuffer();
        int uvRowStride = uPlane.getRowStride();
        int uvPixStride = uPlane.getPixelStride();
        int chromaW = w / 2;
        int chromaH = h / 2;
        for (int row = 0; row < chromaH; row++) {
            int base = row * uvRowStride;
            for (int col = 0; col < chromaW; col++) {
                int idx = base + col * uvPixStride;
                nv21[pos++] = vBuf.get(idx);   // NV21: V first
                nv21[pos++] = uBuf.get(idx);   // then U
            }
        }
        return nv21;
    }

    // ── Face angle detection — same formula as recorder.py / register.py ──────

    // Runs on cameraExecutor. Mirrors recorder.py's _STATE_POSE: detect a face, write
    // (encode) every frame in which one is found, then evaluate the pose and count holds.
    // The imageProxy stays open here (caller closes it) so the frame can be encoded.
    private void onFacesDetected(List<Face> faces, int frameWidth, ImageProxy imageProxy) {
        if (!isRecording || poseState == PoseState.TRANSITIONING) return;

        final int pose = currentPose;

        if (faces.isEmpty()) {
            holdFrameCount = 0;
            runOnUiThread(() -> {
                tvInstruction.setText(POSES[pose]);
                tvCountdown.setText("");
                tvAngleStatus.setText("顔が検出されません");
                tvAngleStatus.setTextColor(0xFFFF4444);
                setProgressBarColor(0xFFFF4444);
                progressBar.setProgress(0);
            });
            return;
        }

        // Select the largest face by bounding-box area — mirrors find_largest_face() in
        // face_processor.py. Prevents background faces from hijacking registration when
        // multiple people are in frame.
        Face face = faces.get(0);
        for (Face f : faces) {
            if (f.getBoundingBox().width() * f.getBoundingBox().height()
                    > face.getBoundingBox().width() * face.getBoundingBox().height()) {
                face = f;
            }
        }

        // Reject faces too small for quality ArcFace embedding — mirrors min_face_size
        // in face_processor.py (the Raspberry Pi/Hailo target). The frame is NOT recorded
        // and pose progression is blocked until the face fills enough of the frame. NOTE:
        // recorder.py's Windows detectors don't enforce this, so this is the one place we
        // intentionally diverge from recorder.py to match the actual Pi pipeline.
        if (face.getBoundingBox().width() < frameWidth * MIN_FACE_FRACTION) {
            holdFrameCount = 0;
            runOnUiThread(() -> {
                tvInstruction.setText(POSES[pose]);
                tvCountdown.setText("");
                tvAngleStatus.setText("もっとカメラに近づいてください");
                tvAngleStatus.setTextColor(0xFFFFAA00);
                setProgressBarColor(0xFFFF4444);
                progressBar.setProgress(0);
            });
            return;
        }

        FaceLandmark leftEye   = face.getLandmark(FaceLandmark.LEFT_EYE);
        FaceLandmark rightEye  = face.getLandmark(FaceLandmark.RIGHT_EYE);
        FaceLandmark leftMouth = face.getLandmark(FaceLandmark.MOUTH_LEFT);
        FaceLandmark rightMouth = face.getLandmark(FaceLandmark.MOUTH_RIGHT);
        // NOSE_BRIDGE contour: starts between eyes, ends at nose tip (last point)
        FaceContour noseBridge = face.getContour(FaceContour.NOSE_BRIDGE);
        List<PointF> bridgePoints = noseBridge != null ? noseBridge.getPoints() : null;

        if (leftEye == null || rightEye == null || leftMouth == null
                || rightMouth == null || bridgePoints == null
                || bridgePoints.isEmpty()) {
            runOnUiThread(() -> {
                tvAngleStatus.setText("ランドマーク検出中...");
                tvAngleStatus.setTextColor(0xFFAAAAAA);
            });
            return;
        }

        // Face detected with usable landmarks → record THIS frame, exactly like
        // recorder.py (recorder.py:621) writes every face-detected frame during the
        // pose-adjustment phase. register.py video-mode uses these in-between frames to
        // cover the full range of head angles, not just the five target poses.
        encodeCurrentFrame(imageProxy);

        PointF noseTip = bridgePoints.get(bridgePoints.size() - 1);

        // Normalized yaw/pitch — identical formula to FaceProcessor in register.py
        float yaw   = estimateYaw(leftEye, rightEye, noseTip);
        float pitch = estimatePitch(leftEye, rightEye, noseTip, leftMouth, rightMouth);

        boolean matches = poseMatches(POSE_KEYS[pose], yaw, pitch);

        if (matches) {
            // Require HOLD_FRAMES consecutive matching frames before starting the
            // fixed-count recording phase — mirrors HOLD_FRAMES=3 in recorder.py.
            holdFrameCount++;
            if (holdFrameCount >= HOLD_FRAMES) {
                // Set the frame count BEFORE flipping to HOLDING. analyzeFrame reads
                // poseState and, on seeing HOLDING, immediately decrements
                // recordingFramesLeft. The volatile write to poseState publishes the
                // already-set recordingFramesLeft; reversing the order would let a stale
                // 0 be decremented and skip the pose with no recording.
                recordingFramesLeft = FRAMES_PER_RECORDING;
                poseState = PoseState.HOLDING;
                holdFrameCount = 0;   // mirrors recorder.py resetting hold_count at this point
                runOnUiThread(() -> {
                    tvInstruction.setText("録画中...");
                    tvCountdown.setText("");   // 録画中の赤丸は非表示
                    tvAngleStatus.setText("角度OK");
                    tvAngleStatus.setTextColor(0xFF00CC66);
                    setProgressBarColor(0xFFFF4444);
                });
            }
            // Once HOLDING, the fixed FRAMES_PER_RECORDING frames are handled by
            // countRecordingFrame() via analyzeFrame.
        } else {
            holdFrameCount = 0;
            String hint = getAngleHint(POSE_KEYS[pose], yaw, pitch);
            int angleProgress = getAngleProgress(POSE_KEYS[pose], yaw, pitch);
            runOnUiThread(() -> {
                tvAngleStatus.setText(hint);
                tvAngleStatus.setTextColor(0xFFFFAA00);
                progressBar.setProgress(angleProgress);
            });
        }
    }

    // Encode the current analysis frame (cameraExecutor only). Used both during the
    // pose-adjustment phase (WAITING) and the fixed-count phase (HOLDING).
    private void encodeCurrentFrame(ImageProxy imageProxy) {
        if (encoder == null) return;
        Bitmap frame;
        try {
            frame = imageProxyToUprightBitmap(imageProxy);
        } catch (Exception e) {
            return;
        }
        if (frame != null) {
            try { encoder.encode(frame); } catch (Exception ignored) {}
            frame.recycle();
        }
    }

    // Identical to FaceProcessor.estimate_yaw_ratio() in face_recognition.
    // Uses nose tip x-position from NOSE_BRIDGE contour — same landmark as SCRFD.
    private float estimateYaw(FaceLandmark leftEye, FaceLandmark rightEye, PointF noseTip) {
        float lx = leftEye.getPosition().x;
        float rx = rightEye.getPosition().x;
        float nx = noseTip.x;
        float eyeMidX = (lx + rx) / 2.0f;
        float interOcular = Math.abs(rx - lx);
        if (interOcular < 1e-3f) return 0.0f;
        return (nx - eyeMidX) / interOcular;
    }

    // Identical to FaceProcessor.estimate_pitch_ratio() in face_recognition.
    // Frontal baseline 0.495 was calibrated for nose tip — same landmark as SCRFD.
    private float estimatePitch(FaceLandmark leftEye, FaceLandmark rightEye, PointF noseTip,
                                 FaceLandmark leftMouth, FaceLandmark rightMouth) {
        float eyeMidY   = (leftEye.getPosition().y + rightEye.getPosition().y) / 2.0f;
        float noseY     = noseTip.y;
        float mouthMidY = (leftMouth.getPosition().y + rightMouth.getPosition().y) / 2.0f;
        float eyeToMouth = mouthMidY - eyeMidY;
        if (Math.abs(eyeToMouth) < 1e-3f) return 0.0f;
        return (noseY - eyeMidY) / eyeToMouth - 0.495f;
    }

    // Identical to _pose_matches() in recorder.py / register.py.
    private boolean poseMatches(String key, float yaw, float pitch) {
        switch (key) {
            case "front": return Math.abs(yaw) <= FRONT_YAW_MAX && Math.abs(pitch) <= FRONT_PITCH_MAX;
            case "right": return yaw <= -SIDE_YAW_MIN;
            case "left":  return yaw >= SIDE_YAW_MIN;
            case "up":    return pitch <= -UPDOWN_PITCH_MIN;
            case "down":  return pitch >= UPDOWN_PITCH_MIN;
            default:      return false;
        }
    }

    private String getAngleHint(String key, float yaw, float pitch) {
        int pct;
        switch (key) {
            case "front":
                return (Math.abs(yaw) > FRONT_YAW_MAX)
                    ? (yaw < 0 ? "少し右に向いています" : "少し左に向いています")
                    : "あごを正面に向けてください";
            case "right":
                pct = (int) Math.max(0, Math.min(99, -yaw / SIDE_YAW_MIN * 100));
                return "右を向いてください (" + pct + "% / yaw:" + String.format("%.2f", yaw) + ")";
            case "left":
                pct = (int) Math.max(0, Math.min(99, yaw / SIDE_YAW_MIN * 100));
                return "左を向いてください (" + pct + "% / yaw:" + String.format("%.2f", yaw) + ")";
            case "up":
                pct = (int) Math.max(0, Math.min(99, -pitch / UPDOWN_PITCH_MIN * 100));
                return "上を向いてください (" + pct + "% / pitch:" + String.format("%.2f", pitch) + ")";
            case "down":
                pct = (int) Math.max(0, Math.min(99, pitch / UPDOWN_PITCH_MIN * 100));
                return "下を向いてください (" + pct + "% / pitch:" + String.format("%.2f", pitch) + ")";
            default:
                return "";
        }
    }

    private int getAngleProgress(String key, float yaw, float pitch) {
        switch (key) {
            case "front": {
                int yp = (int) Math.max(0, 100 - Math.abs(yaw) / FRONT_YAW_MAX * 100);
                int pp = (int) Math.max(0, 100 - Math.abs(pitch) / FRONT_PITCH_MAX * 100);
                return Math.min(yp, pp);
            }
            case "right": return (int) Math.max(0, Math.min(99, -yaw / SIDE_YAW_MIN * 100));
            case "left":  return (int) Math.max(0, Math.min(99,  yaw / SIDE_YAW_MIN * 100));
            case "up":    return (int) Math.max(0, Math.min(99, -pitch / UPDOWN_PITCH_MIN * 100));
            case "down":  return (int) Math.max(0, Math.min(99,  pitch / UPDOWN_PITCH_MIN * 100));
            default:      return 0;
        }
    }

    private void setProgressBarColor(int color) {
        progressBar.setProgressTintList(ColorStateList.valueOf(color));
    }

    // Called from analyzeFrame when poseState == HOLDING.
    // Mirrors _STATE_RECORDING in recorder.py: counts down FRAMES_PER_RECORDING
    // frames without any pose check, then advances to the next pose.
    private void countRecordingFrame() {
        int left = --recordingFramesLeft;
        int written = FRAMES_PER_RECORDING - left;
        int progress = (int) Math.min(100, written * 100 / FRAMES_PER_RECORDING);
        if (left <= 0) {
            poseState = PoseState.TRANSITIONING;
            runOnUiThread(this::nextPose);
        } else {
            runOnUiThread(() -> progressBar.setProgress(progress));
        }
    }

    // ── Wizard flow ───────────────────────────────────────────────────────────

    private void onStartClicked() {
        String name = etName.getText().toString().trim();
        if (name.isEmpty()) {
            etName.setError("名前を入力してください");
            return;
        }
        panelInput.setVisibility(View.GONE);
        startRecording(name);
    }

    private void startRecording(String name) {
        File outputDir = getExternalFilesDir(Environment.DIRECTORY_MOVIES);
        if (outputDir != null) outputDir.mkdirs();

        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        File outputFile = new File(outputDir, name + "_" + timestamp + ".mp4");
        outputFilePath = outputFile.getAbsolutePath();

        try {
            encoder = new Mp4Encoder(outputFilePath, OUT_WIDTH, OUT_HEIGHT,
                ENC_FPS, ENC_BITRATE);
        } catch (IOException e) {
            Toast.makeText(this, "エンコーダの初期化に失敗しました", Toast.LENGTH_LONG).show();
            resetToInput();
            return;
        }

        currentPose = 0;
        isRecording = true;
        panelRecording.setVisibility(View.VISIBLE);
        updatePoseUI();
    }

    private void updatePoseUI() {
        recordingFramesLeft = 0;
        holdFrameCount = 0;
        poseState = PoseState.WAITING;
        tvPoseProgress.setText("[" + (currentPose + 1) + "/" + POSES.length + "]");
        tvInstruction.setText(POSES[currentPose]);
        tvCountdown.setText("");          // 赤丸なし
        tvAngleStatus.setText("顔を検出中...");
        tvAngleStatus.setTextColor(0xFFAAAAAA);
        progressBar.setProgress(0);
        setProgressBarColor(0xFFFF4444);
        playPoseSound(currentPose);
    }

    private void playPoseSound(int poseIndex) {
        if (mediaPlayer != null) {
            mediaPlayer.release();
            mediaPlayer = null;
        }
        mediaPlayer = MediaPlayer.create(this, POSE_RAW_IDS[poseIndex]);
        if (mediaPlayer != null) {
            mediaPlayer.setOnCompletionListener(mp -> {
                mp.release();
                if (mediaPlayer == mp) mediaPlayer = null;
            });
            mediaPlayer.start();
        }
    }

    private void nextPose() {
        currentPose++;
        if (currentPose >= POSES.length) {
            isRecording = false;
            tvInstruction.setText("保存中...");
            tvCountdown.setText("");
            tvAngleStatus.setText("");
            // Finalize the encoder on the camera thread so all encoder access stays on
            // a single thread (encode() also runs there). The single-thread executor
            // guarantees this runs after the last queued frame.
            cameraExecutor.execute(() -> {
                if (encoder != null) {
                    encoder.finish();
                    encoder = null;
                }
                runOnUiThread(() -> showCompletion(outputFilePath));
            });
        } else {
            updatePoseUI();
        }
    }

    // ── Completion ────────────────────────────────────────────────────────────

    private void showCompletion(String filePath) {
        panelRecording.setVisibility(View.GONE);
        panelComplete.setVisibility(View.VISIBLE);
        tvFilePath.setText(filePath);
    }

    private void resetToInput() {
        isRecording = false;
        panelRecording.setVisibility(View.GONE);
        panelInput.setVisibility(View.VISIBLE);
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    @Override
    protected void onDestroy() {
        super.onDestroy();
        isRecording = false;
        if (encoder != null) {
            final Mp4Encoder enc = encoder;
            encoder = null;
            cameraExecutor.execute(enc::finish);
        }
        if (mediaPlayer != null) {
            mediaPlayer.release();
            mediaPlayer = null;
        }
        faceDetector.close();
        cameraExecutor.shutdown();
    }
}
