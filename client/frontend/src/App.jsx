
import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

const API = "http://localhost:3001/api";

export default function App() {
  const [file,   setFile]   = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,  setError]  = useState(null);

  const onDrop = useCallback(accepted => {
    if (accepted[0]) { setFile(accepted[0]); setResult(null); setError(null); }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "audio/*": [".wav", ".mp3", ".flac", ".ogg"] },
    maxFiles: 1
  });

  const analyze = async () => {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);

    const form = new FormData();
    form.append("file", file);

    try {
      const { data } = await axios.post(`${API}/predict`, form);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.error || "Analysis failed. Check API connection.");
    } finally {
      setLoading(false);
    }
  };

  const isFake = result?.label === "fake";
  const barData = result ? [
    { name: "Real",  value: +(result.real_prob * 100).toFixed(1) },
    { name: "Fake",  value: +(result.fake_prob * 100).toFixed(1) },
  ] : [];

  return (
    <div style={{
      minHeight: "100vh", background: "#0f1117", color: "#e2e8f0",
      fontFamily: "system-ui, sans-serif", display: "flex",
      flexDirection: "column", alignItems: "center", padding: "48px 24px"
    }}>
      <h1 style={{ fontSize: 28, fontWeight: 700, marginBottom: 8, color: "#f8fafc" }}>
        Audio Deepfake Detector
      </h1>
      <p style={{ color: "#94a3b8", marginBottom: 40, fontSize: 15 }}>
        Upload a voice recording to detect if it's AI-generated
      </p>

      {/* Dropzone */}
      <div {...getRootProps()} style={{
        width: "100%", maxWidth: 520, border: `2px dashed ${isDragActive ? "#6366f1" : "#334155"}`,
        borderRadius: 16, padding: "40px 24px", textAlign: "center",
        cursor: "pointer", background: isDragActive ? "#1e1b4b" : "#1e293b",
        transition: "all 0.2s", marginBottom: 16
      }}>
        <input {...getInputProps()} />
        <div style={{ fontSize: 40, marginBottom: 12 }}>🎙️</div>
        {file
          ? <p style={{ color: "#a5b4fc", fontWeight: 600 }}>{file.name}</p>
          : <p style={{ color: "#64748b" }}>
              {isDragActive ? "Drop it here" : "Drag & drop audio, or click to browse"}
            </p>
        }
        <p style={{ fontSize: 12, color: "#475569", marginTop: 8 }}>
          Supports .wav .mp3 .flac .ogg — max 10MB
        </p>
      </div>

      {/* Analyze button */}
      <button
        onClick={analyze}
        disabled={!file || loading}
        style={{
          background: (!file || loading) ? "#334155" : "#6366f1",
          color: "#fff", border: "none", borderRadius: 10, padding: "14px 40px",
          fontSize: 16, fontWeight: 600, cursor: (!file || loading) ? "not-allowed" : "pointer",
          marginBottom: 32, transition: "background 0.2s", width: "100%", maxWidth: 520
        }}
      >
        {loading ? "Analyzing..." : "Detect Deepfake"}
      </button>

      {/* Error */}
      {error && (
        <div style={{
          background: "#450a0a", border: "1px solid #991b1b", borderRadius: 10,
          padding: "14px 20px", maxWidth: 520, width: "100%", color: "#fca5a5", marginBottom: 24
        }}>
          {error}
        </div>
      )}

      {/* Result */}
      {result && (
        <div style={{
          background: "#1e293b", borderRadius: 16, padding: 28,
          maxWidth: 520, width: "100%",
          border: `2px solid ${isFake ? "#dc2626" : "#16a34a"}`
        }}>
          {/* Verdict */}
          <div style={{ textAlign: "center", marginBottom: 24 }}>
            <div style={{ fontSize: 52, marginBottom: 8 }}>
              {isFake ? "⚠️" : "✅"}
            </div>
            <div style={{
              fontSize: 28, fontWeight: 800,
              color: isFake ? "#f87171" : "#4ade80"
            }}>
              {isFake ? "FAKE" : "REAL"}
            </div>
            <div style={{ color: "#94a3b8", fontSize: 15, marginTop: 4 }}>
              {result.confidence}% confidence
            </div>
          </div>

          {/* Bar chart */}
          <ResponsiveContainer width="100%" height={120}>
            <BarChart data={barData} barSize={48}>
              <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 13 }} axisLine={false} tickLine={false} />
              <YAxis domain={[0, 100]} tick={{ fill: "#94a3b8", fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip
                formatter={v => [`${v}%`]}
                contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8 }}
              />
              <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                {barData.map((entry, i) => (
                  <Cell key={i} fill={entry.name === "Fake" ? "#ef4444" : "#22c55e"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          {/* Attention visualization */}
          {result.attention?.length > 0 && (
            <div style={{ marginTop: 20 }}>
              <p style={{ color: "#64748b", fontSize: 12, marginBottom: 8 }}>
                Model attention across time (brighter = model focused here)
              </p>
              <div style={{ display: "flex", gap: 2, height: 24, borderRadius: 6, overflow: "hidden" }}>
                {result.attention.map((w, i) => (
                  <div key={i} style={{
                    flex: 1,
                    background: `rgba(99, 102, 241, ${Math.min(w * 4, 1)})`,
                    borderRadius: 2
                  }} />
                ))}
              </div>
            </div>
          )}

          {/* Stats */}
          <div style={{
            display: "grid", gridTemplateColumns: "1fr 1fr",
            gap: 10, marginTop: 20
          }}>
            {[
              ["Real probability", `${(result.real_prob * 100).toFixed(1)}%`],
              ["Fake probability", `${(result.fake_prob * 100).toFixed(1)}%`],
              ["Decision threshold", result.threshold],
              ["Inference time",  `${result.inference_ms}ms`],
            ].map(([k, v]) => (
              <div key={k} style={{
                background: "#0f172a", borderRadius: 8, padding: "10px 14px"
              }}>
                <div style={{ fontSize: 11, color: "#64748b" }}>{k}</div>
                <div style={{ fontSize: 15, fontWeight: 600, color: "#e2e8f0" }}>{v}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}