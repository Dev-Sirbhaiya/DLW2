import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Box, Film, Layers, ChevronRight } from "lucide-react";
import UploadZone from "./components/UploadZone.jsx";
import ModeSelector from "./components/ModeSelector.jsx";
import ProgressPanel from "./components/ProgressPanel.jsx";
import ResultsPanel from "./components/ResultsPanel.jsx";

const STEPS = [
  { id: "upload",    label: "Upload",     icon: Film },
  { id: "configure", label: "Configure",  icon: Layers },
  { id: "process",   label: "Processing", icon: Box },
  { id: "results",   label: "Results",    icon: Layers },
];

export default function App() {
  const [step, setStep] = useState("upload"); // upload | configure | process | results
  const [jobId, setJobId] = useState(null);
  const [mode, setMode] = useState("both");
  const [maxIterations, setMaxIterations] = useState(30000);
  const [numFrames, setNumFrames] = useState(300);

  const stepIndex = STEPS.findIndex((s) => s.id === step);

  const handleUploaded = (id) => {
    setJobId(id);
    setStep("configure");
  };

  const handleStart = async () => {
    setStep("process");
    try {
      await fetch("/api/reconstruct", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jobId, mode, maxIterations, numFrames }),
      });
    } catch (err) {
      console.error("Failed to start reconstruction:", err);
    }
  };

  const handleDone = () => setStep("results");
  const handleReset = () => {
    setStep("upload");
    setJobId(null);
    setMode("both");
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header style={{
        padding: "20px 32px",
        borderBottom: "1px solid rgba(0,212,255,0.08)",
        display: "flex",
        alignItems: "center",
        gap: 16,
        backdropFilter: "blur(12px)",
        background: "rgba(5,10,20,0.6)",
        position: "sticky",
        top: 0,
        zIndex: 100,
      }}>
        <div style={{
          width: 40, height: 40, borderRadius: 10,
          background: "linear-gradient(135deg, #00d4ff, #7b61ff)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 20, boxShadow: "0 0 20px rgba(0,212,255,0.4)",
        }}>
          ✦
        </div>
        <div>
          <h1 style={{
            fontFamily: "var(--font-heading)", fontSize: 20, fontWeight: 700,
            background: "linear-gradient(90deg, #00d4ff, #7b61ff)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            lineHeight: 1.2,
          }}>
            3D Reconstruction Studio
          </h1>
          <p style={{ color: "var(--text-muted)", fontSize: 12 }}>
            Nerfstudio · Splatfacto · 3D Gaussian Splatting
          </p>
        </div>

        {/* Step breadcrumb */}
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 6 }}>
          {STEPS.map((s, i) => {
            const active = s.id === step;
            const done = i < stepIndex;
            return (
              <React.Fragment key={s.id}>
                <div style={{
                  display: "flex", alignItems: "center", gap: 6,
                  opacity: done || active ? 1 : 0.3,
                  transition: "opacity 0.3s",
                }}>
                  <div style={{
                    width: 8, height: 8, borderRadius: "50%",
                    background: active
                      ? "#00d4ff"
                      : done ? "#00ff88" : "rgba(255,255,255,0.2)",
                    boxShadow: active ? "0 0 8px #00d4ff" : "none",
                    transition: "all 0.3s",
                  }} />
                  <span style={{
                    fontSize: 12, fontWeight: active ? 600 : 400,
                    color: active ? "#00d4ff" : done ? "#00ff88" : "var(--text-muted)",
                    display: window.innerWidth > 640 ? "block" : "none",
                  }}>
                    {s.label}
                  </span>
                </div>
                {i < STEPS.length - 1 && (
                  <ChevronRight size={12} color="rgba(255,255,255,0.15)" />
                )}
              </React.Fragment>
            );
          })}
        </div>
      </header>

      {/* ── Main Content ────────────────────────────────────────────────────── */}
      <main style={{ flex: 1, padding: "40px 32px", maxWidth: 1200, margin: "0 auto", width: "100%" }}>
        <AnimatePresence mode="wait">

          {step === "upload" && (
            <motion.div key="upload"
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -24 }}
              transition={{ duration: 0.35 }}
            >
              <Hero />
              <UploadZone onUploaded={handleUploaded} />
            </motion.div>
          )}

          {step === "configure" && (
            <motion.div key="configure"
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -24 }}
              transition={{ duration: 0.35 }}
            >
              <ModeSelector
                mode={mode} setMode={setMode}
                maxIterations={maxIterations} setMaxIterations={setMaxIterations}
                numFrames={numFrames} setNumFrames={setNumFrames}
                onStart={handleStart}
                onBack={() => setStep("upload")}
              />
            </motion.div>
          )}

          {step === "process" && (
            <motion.div key="process"
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -24 }}
              transition={{ duration: 0.35 }}
            >
              <ProgressPanel jobId={jobId} mode={mode} onDone={handleDone} />
            </motion.div>
          )}

          {step === "results" && (
            <motion.div key="results"
              initial={{ opacity: 0, scale: 0.97 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.4 }}
            >
              <ResultsPanel jobId={jobId} mode={mode} onReset={handleReset} />
            </motion.div>
          )}

        </AnimatePresence>
      </main>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <footer style={{
        padding: "16px 32px",
        borderTop: "1px solid rgba(255,255,255,0.04)",
        textAlign: "center",
        color: "var(--text-muted)",
        fontSize: 12,
      }}>
        Nerfstudio · Splatfacto · 3D Gaussian Splatting · Novel View Synthesis
      </footer>
    </div>
  );
}

function Hero() {
  return (
    <div style={{ textAlign: "center", marginBottom: 48 }}>
      <motion.div
        animate={{ y: [0, -8, 0] }}
        transition={{ repeat: Infinity, duration: 4, ease: "easeInOut" }}
        style={{ fontSize: 64, marginBottom: 16, display: "inline-block" }}
      >
        ✦
      </motion.div>
      <h2 style={{
        fontFamily: "var(--font-heading)",
        fontSize: "clamp(28px, 5vw, 48px)",
        fontWeight: 700,
        background: "linear-gradient(135deg, #ffffff 0%, #00d4ff 50%, #7b61ff 100%)",
        WebkitBackgroundClip: "text",
        WebkitTextFillColor: "transparent",
        marginBottom: 12,
        lineHeight: 1.2,
      }}>
        Video → 3D Reconstruction
      </h2>
      <p style={{
        color: "var(--text-secondary)",
        fontSize: "clamp(14px, 2vw, 18px)",
        maxWidth: 560,
        margin: "0 auto",
        lineHeight: 1.7,
      }}>
        Upload a video and generate an interactive <strong style={{ color: "#00d4ff" }}>3D Gaussian Splat</strong> and
        photorealistic <strong style={{ color: "#7b61ff" }}>2D novel-view renders</strong> — powered by Nerfstudio Splatfacto.
      </p>
    </div>
  );
}
