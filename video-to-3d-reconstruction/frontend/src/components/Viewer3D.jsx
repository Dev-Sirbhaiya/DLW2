import React, { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { RotateCcw, ZoomIn, ZoomOut, Maximize2, Info } from "lucide-react";

/**
 * Viewer3D — Renders a Gaussian Splat PLY using gaussian-splats-3d.
 * Falls back to a loading/error card if the library is unavailable.
 */
export default function Viewer3D({ jobId }) {
  const containerRef = useRef(null);
  const viewerRef = useRef(null);
  const [state, setState] = useState("loading"); // loading | ready | error
  const [errorMsg, setErrorMsg] = useState("");
  const [info, setInfo] = useState(null);

  const plyUrl = `/api/results/${jobId}/splat.ply`;

  useEffect(() => {
    let viewer = null;
    let cancelled = false;

    async function init() {
      try {
        // Dynamic import to avoid SSR / build issues
        const GS = await import("gaussian-splats-3d");
        if (cancelled) return;

        // Fetch basic meta
        try {
          const r = await fetch(`/api/results/${jobId}`);
          const d = await r.json();
          if (d.results) setInfo(d.results);
        } catch (_) {}

        if (!containerRef.current) return;

        // Create the viewer
        viewer = new GS.Viewer({
          cameraUp: [0, -1, 0],
          initialCameraPosition: [0, -1, 4],
          initialCameraLookAt: [0, 0, 0],
          sharedMemoryForWorkers: false,
          renderMode: GS.RenderMode.Always,
          selfDrivenMode: true,
          useBuiltInControls: true,
          rootElement: containerRef.current,
        });

        viewerRef.current = viewer;

        await viewer.addSplatScene(plyUrl, {
          splatAlphaRemovalThreshold: 5,
          showLoadingUI: false,
        });

        if (!cancelled) setState("ready");
        viewer.start();
      } catch (err) {
        if (!cancelled) {
          setErrorMsg(err.message || "Failed to load 3D viewer");
          setState("error");
        }
      }
    }

    init();

    return () => {
      cancelled = true;
      try { viewer?.stop?.(); viewer?.dispose?.(); } catch (_) {}
    };
  }, [jobId, plyUrl]);

  const resetCamera = () => {
    try { viewerRef.current?.resetCamera?.(); } catch (_) {}
  };

  return (
    <div style={{ position: "relative", width: "100%", height: "100%", minHeight: 460 }}>
      {/* Three.js canvas container */}
      <div
        ref={containerRef}
        style={{
          width: "100%",
          height: "100%",
          minHeight: 460,
          borderRadius: "var(--radius-lg)",
          overflow: "hidden",
          background: "radial-gradient(ellipse at center, #0a1628 0%, #050a14 100%)",
        }}
      />

      {/* Loading overlay */}
      {state === "loading" && (
        <div style={overlayStyle}>
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
            style={{ fontSize: 40, marginBottom: 12 }}
          >
            ✦
          </motion.div>
          <p style={{ color: "var(--text-secondary)", fontSize: 14 }}>
            Loading 3D Gaussian Splat…
          </p>
          <p style={{ color: "var(--text-muted)", fontSize: 12, marginTop: 6 }}>
            Large scenes may take a moment to stream.
          </p>
        </div>
      )}

      {/* Error overlay */}
      {state === "error" && (
        <div style={overlayStyle}>
          <div style={{ fontSize: 40, marginBottom: 12 }}>❌</div>
          <p style={{ color: "#ff3b5c", fontSize: 15, fontWeight: 600, marginBottom: 6 }}>
            3D Viewer Failed
          </p>
          <p style={{ color: "var(--text-muted)", fontSize: 13, maxWidth: 300, textAlign: "center" }}>
            {errorMsg || "Could not load the Gaussian Splat viewer."}
          </p>
          <a
            href={plyUrl}
            download="splat.ply"
            style={{
              marginTop: 16, padding: "8px 16px",
              background: "rgba(0,212,255,0.1)",
              border: "1px solid rgba(0,212,255,0.3)",
              borderRadius: 8, color: "#00d4ff", fontSize: 13,
              textDecoration: "none",
            }}
          >
            ↓ Download splat.ply
          </a>
        </div>
      )}

      {/* Controls overlay (top-right) */}
      {state === "ready" && (
        <div style={{
          position: "absolute", top: 12, right: 12,
          display: "flex", flexDirection: "column", gap: 6,
        }}>
          {[
            { icon: RotateCcw, title: "Reset camera", action: resetCamera },
          ].map(({ icon: Icon, title, action }) => (
            <button
              key={title}
              onClick={action}
              title={title}
              style={{
                width: 36, height: 36, borderRadius: 8,
                background: "rgba(10,22,40,0.8)",
                border: "1px solid rgba(0,212,255,0.2)",
                color: "var(--text-secondary)",
                display: "flex", alignItems: "center", justifyContent: "center",
                backdropFilter: "blur(8px)",
              }}
            >
              <Icon size={16} />
            </button>
          ))}
        </div>
      )}

      {/* Info badge */}
      {state === "ready" && info?.num_gaussians && (
        <div style={{
          position: "absolute", bottom: 12, left: 12,
          background: "rgba(10,22,40,0.85)",
          border: "1px solid rgba(0,212,255,0.15)",
          borderRadius: 8, padding: "6px 12px",
          display: "flex", alignItems: "center", gap: 6,
          backdropFilter: "blur(8px)",
          fontSize: 12, color: "var(--text-secondary)",
        }}>
          <Info size={12} color="#00d4ff" />
          {Number(info.num_gaussians).toLocaleString()} Gaussians
          {info.size_mb && ` · ${info.size_mb} MB`}
        </div>
      )}

      {/* Controls hint */}
      {state === "ready" && (
        <div style={{
          position: "absolute", bottom: 12, right: 12,
          background: "rgba(10,22,40,0.7)",
          border: "1px solid rgba(255,255,255,0.06)",
          borderRadius: 8, padding: "6px 10px",
          fontSize: 11, color: "var(--text-muted)",
          backdropFilter: "blur(8px)",
        }}>
          Drag to orbit · Scroll to zoom · Right-drag to pan
        </div>
      )}
    </div>
  );
}

const overlayStyle = {
  position: "absolute", inset: 0,
  display: "flex", flexDirection: "column",
  alignItems: "center", justifyContent: "center",
  background: "rgba(5,10,20,0.75)",
  borderRadius: "var(--radius-lg)",
  backdropFilter: "blur(8px)",
};
