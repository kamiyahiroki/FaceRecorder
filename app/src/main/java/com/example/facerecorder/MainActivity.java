package com.example.facerecorder;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
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
import androidx.camera.core.Preview;
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

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.ExecutionException;

public class MainActivity extends AppCompatActivity {

    private static final int COUNTDOWN_SEC = 3;
    private static final int RECORD_SEC = 4;

    private static final String[] POSES = {
        "正面を向いてください",
        "ゆっくり右を向いてください",
        "ゆっくり左を向いてください",
        "ゆっくり上を向いてください",
        "ゆっくり下を向いてください"
    };

    // Views
    private PreviewView previewView;
    private View panelInput;
    private View panelRecording;
    private View panelComplete;
    private EditText etName;
    private TextView tvPoseProgress;
    private TextView tvInstruction;
    private TextView tvCountdown;
    private ProgressBar progressBar;
    private TextView tvFilePath;

    // CameraX
    private VideoCapture<Recorder> videoCapture;
    private Recording activeRecording;

    // Wizard state
    private int currentPose = 0;
    private final Handler handler = new Handler(Looper.getMainLooper());
    private Runnable pendingRunnable;

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
        progressBar    = findViewById(R.id.progressBar);
        tvFilePath     = findViewById(R.id.tvFilePath);

        Button btnStart = findViewById(R.id.btnStart);
        Button btnClose = findViewById(R.id.btnClose);
        btnStart.setOnClickListener(v -> onStartClicked());
        btnClose.setOnClickListener(v -> finish());

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
            .setQualitySelector(QualitySelector.from(Quality.HD))
            .build();
        videoCapture = VideoCapture.withOutput(recorder);

        // Prefer front camera; fall back to back if unavailable
        CameraSelector selector;
        try {
            selector = provider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA)
                ? CameraSelector.DEFAULT_FRONT_CAMERA
                : CameraSelector.DEFAULT_BACK_CAMERA;
        } catch (androidx.camera.core.CameraInfoUnavailableException e) {
            selector = CameraSelector.DEFAULT_BACK_CAMERA;
        }

        provider.bindToLifecycle(this, selector, preview, videoCapture);
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
        if (outputDir != null) {
            outputDir.mkdirs();
        }
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
        panelRecording.setVisibility(View.VISIBLE);
        startCountdown();
    }

    private void startCountdown() {
        tvPoseProgress.setText("[" + (currentPose + 1) + "/" + POSES.length + "]");
        tvInstruction.setText(POSES[currentPose]);
        tvCountdown.setTextColor(0xFFFFFFFF);
        progressBar.setProgress(0);
        runCountdown(COUNTDOWN_SEC);
    }

    private void runCountdown(int count) {
        if (count > 0) {
            tvCountdown.setText(String.valueOf(count));
            pendingRunnable = () -> runCountdown(count - 1);
            handler.postDelayed(pendingRunnable, 1000);
        } else {
            startPoseRecording();
        }
    }

    private void startPoseRecording() {
        tvCountdown.setText("●");
        tvCountdown.setTextColor(0xFFFF4444);
        progressBar.setProgress(0);

        final long startTime = System.currentTimeMillis();
        final long durationMs = RECORD_SEC * 1000L;

        pendingRunnable = new Runnable() {
            @Override
            public void run() {
                long elapsed = System.currentTimeMillis() - startTime;
                progressBar.setProgress((int) (elapsed * 100 / durationMs));
                if (elapsed < durationMs) {
                    handler.postDelayed(this, 50);
                } else {
                    progressBar.setProgress(100);
                    nextPose();
                }
            }
        };
        handler.postDelayed(pendingRunnable, 50);
    }

    private void nextPose() {
        currentPose++;
        if (currentPose >= POSES.length) {
            tvInstruction.setText("保存中...");
            tvCountdown.setText("");
            if (activeRecording != null) {
                activeRecording.stop();
                activeRecording = null;
            }
        } else {
            startCountdown();
        }
    }

    // ── Completion ────────────────────────────────────────────────────────────

    private void showCompletion(String filePath) {
        panelRecording.setVisibility(View.GONE);
        panelComplete.setVisibility(View.VISIBLE);
        tvFilePath.setText(filePath);
    }

    private void resetToInput() {
        panelRecording.setVisibility(View.GONE);
        panelInput.setVisibility(View.VISIBLE);
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    @Override
    protected void onDestroy() {
        super.onDestroy();
        handler.removeCallbacksAndMessages(null);
        if (activeRecording != null) {
            activeRecording.stop();
            activeRecording = null;
        }
    }
}
