import React, { useRef, useState } from "react";
import { motion } from "framer-motion";
import { Play, Pause, Volume2, VolumeX, Maximize2, Image } from "lucide-react";

export default function Viewer2D({ jobId }) {
  const videoRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(true);
  const [showSheet, setShowSheet] = useState(false);

  const mp4Url   = `/api/results/${jobId}/render.mp4`;
  const sheetUrl = `/api/results/${jobId}/contact_sheet.png`;

  const toggle = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) { v.play(); setPlaying(true); }
    else          { v.pause(); setPlaying(false); }
  };

  const toggleMute = () => {
    const v = videoRef.current;
    if (!v) return;
    v.muted = !v.muted;
    setMuted(v.muted);
  };

  const fullscreen = () => videoRef.current?.requestFullscreen?.();

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, height: "100%" }}>
      {/* Video player */}
      <div style={{
        position: "relative",
        background: "#000",
        borderRadius: "var(--radius-lg)",
        overflow: "hidden",
        flex: 1,
        minHeight: 300,
      }}>
        <video
          ref={videoRef}
          src={mp4Url}
          loop
          muted
          playsInline
          style={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
        />

        {/* Controls bar */}
        <div style={{
          position: "absolute", bottom: 0, left: 0, right: 0,
          background: "linear-gradient(transparent, rgba(0,0,0,0.8))",
          padding: "20px 14px 12px",
          display: "flex", alignItems: "center", gap: 10,
        }}>
          <button onClick={toggle} style={ctrlBtn}>
            {playing ? <Pause size={16} /> : <Play size={16} />}
          </button>

          <button onClick={toggleMute} style={ctrlBtn}>
            {muted ? <VolumeX size={16} /> : <Volume2 size={16} />}
          </button>

          <span style={{ flex: 1 }} />

          <button
            onClick={() => setShowSheet((v) => !v)}
            title="Frame contact sheet"
            style={{
              ...ctrlBtn,
              background: showSheet ? "rgba(0,212,255,0.2)" : "rgba(255,255,255,0.1)",
              border: showSheet ? "1px solid rgba(0,212,255,0.4)" : "1px solid transparent",
            }}
          >
            <Image size={16} />
          </button>

          <button onClick={fullscreen} style={ctrlBtn}>
            <Maximize2 size={16} />
          </button>
        </div>

        {/* Big play button when paused */}
        {!playing && (
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            onClick={toggle}
            style={{
              position: "absolute", inset: 0, margin: "auto",
              width: 64, height: 64, borderRadius: "50%",
              background: "rgba(0,212,255,0.2)",
              border: "2px solid rgba(0,212,255,0.5)",
              display: "flex", alignItems: "center", justifyContent: "center",
              backdropFilter: "blur(8px)",
              color: "#00d4ff",
            }}
          >
            <Play size={28} fill="#00d4ff" />
          </motion.button>
        )}
      </div>

      {/* Contact sheet */}
      {showSheet && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          exit={{ opacity: 0, height: 0 }}
          style={{
            borderRadius: "var(--radius-md)",
            overflow: "hidden",
            border: "1px solid rgba(0,212,255,0.15)",
            background: "#050a14",
          }}
        >
          <div style={{
            padding: "8px 14px",
            background: "rgba(0,212,255,0.06)",
            borderBottom: "1px solid rgba(0,212,255,0.1)",
            fontSize: 12, color: "var(--text-muted)", fontWeight: 500,
          }}>
            <Image size={12} style={{ display: "inline", marginRight: 6 }} />
            Sampled Render Frames
          </div>
          <img
            src={sheetUrl}
            alt="Contact sheet of rendered frames"
            style={{ width: "100%", display: "block" }}
            onError={(e) => { e.target.style.display = "none"; }}
          />
        </motion.div>
      )}

      {/* Download */}
      <a
        href={mp4Url}
        download="render.mp4"
        style={{
          textAlign: "center",
          padding: "10px",
          background: "rgba(123,97,255,0.08)",
          border: "1px solid rgba(123,97,255,0.2)",
          borderRadius: 10,
          color: "#7b61ff",
          fontSize: 13, fontWeight: 500,
          textDecoration: "none",
          display: "block",
        }}
      >
        ↓ Download render.mp4
      </a>
    </div>
  );
}

const ctrlBtn = {
  width: 34, height: 34, borderRadius: 8,
  background: "rgba(255,255,255,0.1)",
  border: "1px solid transparent",
  color: "#fff",
  display: "flex", alignItems: "center", justifyContent: "center",
  cursor: "pointer",
};
