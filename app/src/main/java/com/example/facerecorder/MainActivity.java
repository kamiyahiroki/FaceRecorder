package com.example.facerecorder;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.content.res.ColorStateList;
import android.graphics.PointF;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.Environment;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Preview;
import androidx.camera.core.MirrorMode;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.video.FileOutputOptions;
import androidx.camera.video.Quality;
import androidx.camera.video.QualitySelector;
import androidx.camera.video.Recorder;
import androidx.camera.video.Recording;
import androidx.camera.video.VideoCapture;
import androidx.camera.video.VideoRecordEvent;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceContour;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.google.mlkit.vision.face.FaceLandmark;

import java.io.File;
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
    // register.py uses UPDOWN_PITCH_MIN = 0.12 for final acceptance.
    // ML Kit's NOSE_BRIDGE endpoint and MOUTH landmarks sit at slightly different
    // positions from SCRFD's 5-point output, causing ML Kit pitch to be
    // consistently higher (more positive) than SCRFD for the same physical tilt.
    // Empirically this makes FaceRecorder accept "down" too early — the user
    // reaches ML Kit pitch=0.12 while SCRFD pitch is still ~0.07–0.09.
    // Setting UPDOWN_PITCH_MIN to 0.17 compensates: when ML Kit says OK the user
    // has tilted far enough that SCRFD also measures ≥ 0.12.
    private static final float FRONT_YAW_MAX    = 0.12f;
    private static final float SIDE_YAW_MIN     = 0.18f;
    private static final float FRONT_PITCH_MAX  = 0.06f;
    private static final float UPDOWN_PITCH_MIN = 0.17f;  // register.py uses 0.12; +0.05 margin for ML Kit offset

    // How long (ms) the user must hold the target pose before advancing.
    // face_recognition needs 10 frames at ~30fps ≈ 0.33s; 2s gives a safe margin.
    private static final int HOLD_DURATION_MS = 2000;

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
    private View panelInput, panelRecording, panelComplete;
    private EditText etName;
    private TextView tvPoseProgress, tvInstruction, tvCountdown, tvAngleStatus, tvFilePath;
    private ProgressBar progressBar;

    // Audio
    private MediaPlayer mediaPlayer;

    // CameraX
    private VideoCapture<Recorder> videoCapture;
    private Recording activeRecording;
    private FaceDetector faceDetector;
    private ExecutorService cameraExecutor;

    // Wizard state
    private int currentPose = 0;
    private volatile PoseState poseState = PoseState.WAITING;
    private volatile long holdStartMs = 0;
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

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    // ── Camera setup ──────────────────────────────────────────────────────────

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

        Recorder recorder = new Recorder.Builder()
            .setQualitySelector(QualitySelector.from(Quality.SD))
            .build();
        videoCapture = new VideoCapture.Builder<>(recorder)
            .setMirrorMode(MirrorMode.MIRROR_MODE_OFF)
            .build();

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
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

        provider.bindToLifecycle(this, selector, preview, videoCapture, imageAnalysis);
    }

    @SuppressLint("UnsafeOptInUsageError")
    private void analyzeFrame(androidx.camera.core.ImageProxy imageProxy) {
        if (!isRecording || imageProxy.getImage() == null) {
            imageProxy.close();
            return;
        }
        InputImage image = InputImage.fromMediaImage(
            imageProxy.getImage(), imageProxy.getImageInfo().getRotationDegrees());
        faceDetector.process(image)
            .addOnSuccessListener(this::onFacesDetected)
            .addOnCompleteListener(t -> imageProxy.close());
    }

    // ── Face angle detection — same formula as recorder.py / register.py ──────

    private void onFacesDetected(List<Face> faces) {
        if (!isRecording || poseState == PoseState.TRANSITIONING) return;

        if (faces.isEmpty()) {
            if (poseState == PoseState.HOLDING) poseState = PoseState.WAITING;
            runOnUiThread(() -> {
                tvInstruction.setText(POSES[currentPose]);
                tvCountdown.setText("");
                tvAngleStatus.setText("顔が検出されません");
                tvAngleStatus.setTextColor(0xFFFF4444);
                setProgressBarColor(0xFFFF4444);
                progressBar.setProgress(0);
            });
            return;
        }

        Face face = faces.get(0);

        FaceLandmark leftEye   = face.getLandmark(FaceLandmark.LEFT_EYE);
        FaceLandmark rightEye  = face.getLandmark(FaceLandmark.RIGHT_EYE);
        FaceLandmark leftMouth = face.getLandmark(FaceLandmark.MOUTH_LEFT);
        FaceLandmark rightMouth = face.getLandmark(FaceLandmark.MOUTH_RIGHT);
        // NOSE_BRIDGE contour: starts between eyes, ends at nose tip (last point)
        FaceContour noseBridge = face.getContour(FaceContour.NOSE_BRIDGE);

        if (leftEye == null || rightEye == null || leftMouth == null
                || rightMouth == null || noseBridge == null
                || noseBridge.getPoints().isEmpty()) {
            runOnUiThread(() -> {
                tvAngleStatus.setText("ランドマーク検出中...");
                tvAngleStatus.setTextColor(0xFFAAAAAA);
            });
            return;
        }

        List<PointF> bridgePoints = noseBridge.getPoints();
        PointF noseTip = bridgePoints.get(bridgePoints.size() - 1);

        // Normalized yaw/pitch — identical formula to FaceProcessor in register.py
        float yaw   = estimateYaw(leftEye, rightEye, noseTip);
        float pitch = estimatePitch(leftEye, rightEye, noseTip, leftMouth, rightMouth);

        boolean matches = poseMatches(POSE_KEYS[currentPose], yaw, pitch);

        if (matches) {
            if (poseState == PoseState.WAITING) {
                poseState = PoseState.HOLDING;
                holdStartMs = System.currentTimeMillis();
                runOnUiThread(() -> {
                    tvInstruction.setText("✓ そのままキープ！");
                    tvCountdown.setText("✓");
                    tvCountdown.setTextColor(0xFF00CC66);
                    tvAngleStatus.setText("角度OK");
                    tvAngleStatus.setTextColor(0xFF00CC66);
                    setProgressBarColor(0xFF00CC66);
                });
            }
            long elapsed = System.currentTimeMillis() - holdStartMs;
            int progress = (int) Math.min(100, elapsed * 100 / HOLD_DURATION_MS);
            if (elapsed >= HOLD_DURATION_MS) {
                poseState = PoseState.TRANSITIONING;
                runOnUiThread(this::nextPose);
            } else {
                runOnUiThread(() -> progressBar.setProgress(progress));
            }
        } else {
            if (poseState == PoseState.HOLDING) {
                poseState = PoseState.WAITING;
                runOnUiThread(() -> {
                    tvInstruction.setText(POSES[currentPose]);
                    tvCountdown.setText("");
                    setProgressBarColor(0xFFFF4444);
                });
            }
            String hint = getAngleHint(POSE_KEYS[currentPose], yaw, pitch);
            int angleProgress = getAngleProgress(POSE_KEYS[currentPose], yaw, pitch);
            runOnUiThread(() -> {
                tvAngleStatus.setText(hint);
                tvAngleStatus.setTextColor(0xFFFFAA00);
                progressBar.setProgress(angleProgress);
            });
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

        FileOutputOptions options = new FileOutputOptions.Builder(outputFile).build();

        activeRecording = videoCapture.getOutput()
            .prepareRecording(this, options)
            .start(ContextCompat.getMainExecutor(this), event -> {
                if (event instanceof VideoRecordEvent.Finalize) {
                    VideoRecordEvent.Finalize fin = (VideoRecordEvent.Finalize) event;
                    if (!fin.hasError()) {
                        showCompletion(outputFile.getAbsolutePath());
                    } else {
                        Toast.makeText(this, "録画エラー: " + fin.getError(),
                            Toast.LENGTH_LONG).show();
                        resetToInput();
                    }
                }
            });

        currentPose = 0;
        isRecording = true;
        poseState = PoseState.WAITING;
        panelRecording.setVisibility(View.VISIBLE);
        updatePoseUI();
    }

    private void updatePoseUI() {
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
            if (activeRecording != null) {
                activeRecording.stop();
                activeRecording = null;
            }
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
        if (activeRecording != null) {
            activeRecording.stop();
            activeRecording = null;
        }
        if (mediaPlayer != null) {
            mediaPlayer.release();
            mediaPlayer = null;
        }
        faceDetector.close();
        cameraExecutor.shutdown();
    }
}
