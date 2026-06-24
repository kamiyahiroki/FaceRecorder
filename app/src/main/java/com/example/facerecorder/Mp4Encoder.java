package com.example.facerecorder;

import android.graphics.Bitmap;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaCodecList;
import android.media.MediaFormat;
import android.media.MediaMuxer;

import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * Minimal H.264/MP4 encoder that takes already-upright {@link Bitmap} frames and
 * muxes them into a single .mp4 at a fixed resolution.
 *
 * Replaces CameraX {@code VideoCapture<Recorder>}, which can only pick logical
 * Quality presets (SD/HD/…) whose pixel size is device-dependent. Here the output
 * is guaranteed to be exactly {@code width}×{@code height} because every frame is
 * scaled to that size before encoding.
 *
 * Threading: not thread-safe. Construct on any thread, but call {@link #encode}
 * and {@link #finish} from a single thread (the camera analysis thread here).
 */
public class Mp4Encoder {

    private static final String MIME = "video/avc";

    private final int width;
    private final int height;
    private final int fps;
    private final int colorFormat;
    private final boolean semiPlanar;

    private final MediaCodec codec;
    private final MediaMuxer muxer;
    private final MediaCodec.BufferInfo bufferInfo = new MediaCodec.BufferInfo();
    private final byte[] yuvBuf;
    private final int[] argbBuf;

    private int trackIndex = -1;
    private boolean muxerStarted = false;
    private long frameIndex = 0;

    public Mp4Encoder(String outputPath, int width, int height, int fps, int bitRate)
            throws IOException {
        this.width = width;
        this.height = height;
        this.fps = fps;

        MediaCodecInfo codecInfo = selectCodec();
        if (codecInfo == null) throw new IOException("No AVC encoder available");
        this.colorFormat = selectColorFormat(codecInfo);
        this.semiPlanar =
            colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar;

        MediaFormat format = MediaFormat.createVideoFormat(MIME, width, height);
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT, colorFormat);
        format.setInteger(MediaFormat.KEY_BIT_RATE, bitRate);
        format.setInteger(MediaFormat.KEY_FRAME_RATE, fps);
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1);

        codec = MediaCodec.createByCodecName(codecInfo.getName());
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
        codec.start();

        muxer = new MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);

        yuvBuf = new byte[width * height * 3 / 2];
        argbBuf = new int[width * height];
    }

    /** Encode one frame. The bitmap is scaled to width×height if it is not already. */
    public void encode(Bitmap frame) {
        Bitmap scaled = (frame.getWidth() == width && frame.getHeight() == height)
            ? frame
            : Bitmap.createScaledBitmap(frame, width, height, true);

        bitmapToYuv(scaled);
        if (scaled != frame) scaled.recycle();

        long ptsUs = frameIndex * 1_000_000L / fps;
        int inIndex = codec.dequeueInputBuffer(10_000);
        if (inIndex >= 0) {
            ByteBuffer input = codec.getInputBuffer(inIndex);
            input.clear();
            input.put(yuvBuf);
            codec.queueInputBuffer(inIndex, 0, yuvBuf.length, ptsUs, 0);
            frameIndex++;
        }
        drain(false);
    }

    /** Flush, finalize the .mp4, and release all resources. */
    public void finish() {
        try {
            int inIndex;
            do {
                inIndex = codec.dequeueInputBuffer(10_000);
            } while (inIndex < 0);
            long ptsUs = frameIndex * 1_000_000L / fps;
            codec.queueInputBuffer(inIndex, 0, 0, ptsUs,
                MediaCodec.BUFFER_FLAG_END_OF_STREAM);
            drain(true);
        } catch (Exception ignored) {
            // best-effort flush; release below regardless
        }
        release();
    }

    private void drain(boolean endOfStream) {
        while (true) {
            int outIndex = codec.dequeueOutputBuffer(bufferInfo, 10_000);
            if (outIndex == MediaCodec.INFO_TRY_AGAIN_LATER) {
                if (!endOfStream) return;     // no more output for now
                continue;                     // keep waiting for EOS
            } else if (outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                if (!muxerStarted) {
                    trackIndex = muxer.addTrack(codec.getOutputFormat());
                    muxer.start();
                    muxerStarted = true;
                }
            } else if (outIndex >= 0) {
                ByteBuffer encoded = codec.getOutputBuffer(outIndex);
                if ((bufferInfo.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) {
                    bufferInfo.size = 0;      // codec config consumed by addTrack()
                }
                if (bufferInfo.size > 0 && muxerStarted && encoded != null) {
                    encoded.position(bufferInfo.offset);
                    encoded.limit(bufferInfo.offset + bufferInfo.size);
                    muxer.writeSampleData(trackIndex, encoded, bufferInfo);
                }
                codec.releaseOutputBuffer(outIndex, false);
                if ((bufferInfo.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) return;
            }
        }
    }

    private void release() {
        try { codec.stop(); } catch (Exception ignored) {}
        try { codec.release(); } catch (Exception ignored) {}
        try { if (muxerStarted) muxer.stop(); } catch (Exception ignored) {}
        try { muxer.release(); } catch (Exception ignored) {}
    }

    // ── RGB → YUV420 (BT.601 video range) ───────────────────────────────────────
    // Fills yuvBuf as NV12 (semi-planar, U/V interleaved) or I420 (planar),
    // matching the encoder's negotiated color format.
    private void bitmapToYuv(Bitmap bmp) {
        bmp.getPixels(argbBuf, 0, width, 0, 0, width, height);

        final int frameSize = width * height;
        int yIndex = 0;
        int uvIndex = frameSize;                 // NV12 interleaved base
        int uIndex = frameSize;                  // I420 U plane base
        int vIndex = frameSize + frameSize / 4;  // I420 V plane base

        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                int p = argbBuf[j * width + i];
                int r = (p >> 16) & 0xff;
                int g = (p >> 8) & 0xff;
                int b = p & 0xff;

                int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
                yuvBuf[yIndex++] = (byte) clamp(y);

                // Sub-sample chroma at every 2×2 block (top-left sample).
                if ((j & 1) == 0 && (i & 1) == 0) {
                    int u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                    int v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
                    if (semiPlanar) {
                        yuvBuf[uvIndex++] = (byte) clamp(u);
                        yuvBuf[uvIndex++] = (byte) clamp(v);
                    } else {
                        yuvBuf[uIndex++] = (byte) clamp(u);
                        yuvBuf[vIndex++] = (byte) clamp(v);
                    }
                }
            }
        }
    }

    private static int clamp(int v) {
        return v < 0 ? 0 : (v > 255 ? 255 : v);
    }

    private static MediaCodecInfo selectCodec() {
        MediaCodecList list = new MediaCodecList(MediaCodecList.REGULAR_CODECS);
        for (MediaCodecInfo info : list.getCodecInfos()) {
            if (!info.isEncoder()) continue;
            for (String type : info.getSupportedTypes()) {
                if (type.equalsIgnoreCase(MIME)) return info;
            }
        }
        return null;
    }

    private static int selectColorFormat(MediaCodecInfo info) {
        MediaCodecInfo.CodecCapabilities caps = info.getCapabilitiesForType(MIME);
        for (int cf : caps.colorFormats) {
            if (cf == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar) return cf;
        }
        for (int cf : caps.colorFormats) {
            if (cf == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar) return cf;
        }
        // Fall back to the most widely supported layout if neither is advertised.
        return MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar;
    }
}
