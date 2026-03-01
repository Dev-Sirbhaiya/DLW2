import React, { useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, Film, CheckCircle, AlertCircle, X } from "lucide-react";
import axios from "axios";

const ACCEPTED = ".mp4,.mov,.avi,.mkv,.webm,.m4v";
const MAX_MB = 2000;

export default function UploadZone({ onUploaded }) {
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const handleFile = useCallback((f) => {
    setError(null);
    if (!f) return;

    const sizeMB = f.size / 1e6;
    if (sizeMB > MAX_MB) {
      setError(`File too large (${sizeMB.toFixed(0)} MB). Maximum is ${MAX_MB} MB.`);
      return;
    }

    setFile(f);
    const url = URL.createObjectURL(f);
    setPreview(url);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const onDragOver = (e) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = () => setDragging(false);

  const onInputChange = (e) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    setProgress(0);

    const form = new FormData();
    form.append("video", file);

    try {
      const { data } = await axios.post("/api/upload", form, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (e) => {
          if (e.total) setProgress(Math.round((e.loaded / e.total) * 100));
        },
      });
      onUploaded(data.jobId);
    } catch (err) {
      setError(err.response?.data?.error || err.message || "Upload failed.");
      setUploading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setError(null);
    setProgress(0);
    setUploading(false);
    if (preview) URL.revokeObjectURL(preview);
  };

  return (
    <div style={{ maxWidth: 640, margin: "0 auto" }}>
      <AnimatePresence mode="wait">
        {!file ? (
          /* ── Drop Zone ── */
          <motion.div
            key="dropzone"
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.96 }}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onClick={() => inputRef.current?.click()}
            style={{
              border: `2px dashed ${dragging ? "rgba(0,212,255,0.7)" : "rgba(0,212,255,0.2)"}`,
              borderRadius: "var(--radius-xl)",
              background: dragging
                ? "rgba(0,212,255,0.05)"
                : "rgba(10,22,40,0.5)",
              padding: "64px 40px",
              textAlign: "center",
              cursor: "pointer",
              transition: "all 0.25s ease",
              backdropFilter: "blur(12px)",
              boxShadow: dragging ? "0 0 40px rgba(0,212,255,0.15)" : "none",
            }}
          >
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED}
              onChange={onInputChange}
              style={{ display: "none" }}
            />

            <motion.div
              animate={{ scale: dragging ? 1.15 : 1 }}
              style={{
                width: 72, height: 72, borderRadius: 20,
                background: "linear-gradient(135deg, rgba(0,212,255,0.15), rgba(123,97,255,0.15))",
                border: "1px solid rgba(0,212,255,0.2)",
                display: "flex", alignItems: "center", justifyContent: "center",
                margin: "0 auto 20px",
              }}
            >
              <Upload size={32} color={dragging ? "#00d4ff" : "#8b9ab5"} />
            </motion.div>

            <p style={{ fontSize: 18, fontWeight: 600, color: "var(--text-primary)", marginBottom: 8 }}>
              {dragging ? "Drop it here!" : "Drag & drop your video"}
            </p>
            <p style={{ color: "var(--text-secondary)", fontSize: 14, marginBottom: 16 }}>
              or click to browse
            </p>
            <p style={{
              display: "inline-block",
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 8, padding: "4px 12px",
              fontSize: 12, color: "var(--text-muted)",
            }}>
              MP4 · MOV · AVI · MKV · WebM · up to {MAX_MB} MB
            </p>
          </motion.div>
        ) : (
          /* ── File Preview ── */
          <motion.div
            key="preview"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -16 }}
            className="glass"
            style={{ padding: 28, position: "relative" }}
          >
            {/* Remove button */}
            {!uploading && (
              <button onClick={reset} style={{
                position: "absolute", top: 16, right: 16,
                background: "rgba(255,59,92,0.15)",
                border: "1px solid rgba(255,59,92,0.3)",
                borderRadius: 8, padding: "4px 8px",
                color: "#ff3b5c", display: "flex", alignItems: "center", gap: 4,
                fontSize: 13,
              }}>
                <X size={14} /> Remove
              </button>
            )}

            {/* Video preview */}
            <div style={{ display: "flex", gap: 20, alignItems: "flex-start", marginBottom: 24 }}>
              <div style={{
                width: 120, height: 68, borderRadius: 10, overflow: "hidden",
                border: "1px solid rgba(0,212,255,0.15)", flexShrink: 0,
                background: "#000",
              }}>
                <video
                  src={preview}
                  style={{ width: "100%", height: "100%", objectFit: "cover" }}
                  muted
                />
              </div>
              <div style={{ minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                  <Film size={16} color="#00d4ff" />
                  <span style={{
                    fontSize: 15, fontWeight: 600, color: "var(--text-primary)",
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }}>
                    {file.name}
                  </span>
                </div>
                <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
                  {(file.size / 1e6).toFixed(1)} MB · {file.type || "video"}
                </p>
              </div>
            </div>

            {/* Upload progress */}
            {uploading && (
              <div style={{ marginBottom: 20 }}>
                <div style={{
                  height: 6, borderRadius: 3,
                  background: "rgba(255,255,255,0.08)",
                  overflow: "hidden", marginBottom: 8,
                }}>
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                    style={{
                      height: "100%", borderRadius: 3,
                      background: "linear-gradient(90deg, #00d4ff, #7b61ff)",
                    }}
                  />
                </div>
                <p style={{ color: "var(--text-secondary)", fontSize: 13, textAlign: "right" }}>
                  Uploading… {progress}%
                </p>
              </div>
            )}

            {/* Error */}
            {error && (
              <div style={{
                display: "flex", alignItems: "center", gap: 8,
                background: "rgba(255,59,92,0.1)",
                border: "1px solid rgba(255,59,92,0.3)",
                borderRadius: 10, padding: "10px 14px",
                marginBottom: 16, color: "#ff3b5c", fontSize: 14,
              }}>
                <AlertCircle size={16} />
                {error}
              </div>
            )}

            {/* Upload button */}
            {!uploading && (
              <UploadBtn onClick={handleUpload} />
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function UploadBtn({ onClick }) {
  return (
    <motion.button
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      style={{
        width: "100%", padding: "14px 24px",
        borderRadius: 12, fontWeight: 600, fontSize: 15,
        background: "linear-gradient(135deg, #00d4ff, #7b61ff)",
        color: "#fff",
        boxShadow: "0 4px 20px rgba(0,212,255,0.3)",
        display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
      }}
    >
      <Upload size={18} />
      Upload &amp; Continue
    </motion.button>
  );
}
