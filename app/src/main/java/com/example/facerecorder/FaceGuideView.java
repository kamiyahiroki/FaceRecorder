package com.example.facerecorder;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

/**
 * Semi-transparent face position guide overlay.
 *
 * Darkens the area outside an oval that marks where the user's head should be,
 * mirroring _draw_face_guide() in recorder.py.
 *
 * Ellipse proportions match recorder.py's 640×480 reference frame:
 *   center (320, 230) → cx=0.500, cy=0.479
 *   radii  (155, 185) → rx=0.242, ry=0.385
 */
public class FaceGuideView extends View {

    // This view fills the 4:3 camera band, which maps 1:1 to the recorded 640×480
    // frame, so the proportions match recorder.py's 640×480 reference directly.
    private static final float CX_RATIO = 0.500f;
    private static final float CY_RATIO = 0.479f;
    private static final float RX_RATIO = 0.242f;
    private static final float RY_RATIO = 0.385f;

    private final Paint overlayPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Path   overlayPath = new Path();
    private final RectF  ovalRect    = new RectF();

    public FaceGuideView(Context context) {
        super(context); init();
    }
    public FaceGuideView(Context context, AttributeSet attrs) {
        super(context, attrs); init();
    }
    public FaceGuideView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr); init();
    }

    private void init() {
        // ~55% dark overlay — matches recorder.py's frame * 0.45 darkening
        overlayPaint.setColor(0x8C000000);
        overlayPaint.setStyle(Paint.Style.FILL);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float w = getWidth();
        float h = getHeight();
        if (w == 0 || h == 0) return;

        float cx = w * CX_RATIO;
        float cy = h * CY_RATIO;
        float rx = w * RX_RATIO;
        float ry = h * RY_RATIO;
        ovalRect.set(cx - rx, cy - ry, cx + rx, cy + ry);

        // EVEN_ODD fill: the oval region is "subtracted" from the full-screen rect,
        // leaving the oval transparent and everything outside darkened.
        overlayPath.reset();
        overlayPath.setFillType(Path.FillType.EVEN_ODD);
        overlayPath.addRect(0, 0, w, h, Path.Direction.CW);
        overlayPath.addOval(ovalRect, Path.Direction.CW);
        canvas.drawPath(overlayPath, overlayPaint);
    }
}
