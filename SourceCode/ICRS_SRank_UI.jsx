import { useState, useCallback } from "react";

const NAVY = "#1E3763";
const ORANGE = "#E87621";
const LIGHT_BG = "#F7F6F3";
const CARD_BG = "#FFFFFF";

const MOCK_RESULT = {
  success: true,
  message: "Successfully ranked 3 candidates.",
  job_description: {
    title: "Senior Data Scientist",
    required_skills: ["python", "machine learning", "tensorflow", "sql", "aws"],
    min_experience: 5,
    required_education: "Masters"
  },
  candidates: [
    {
      rank: 1, name: "Priya Sharma", email: "priya@email.com", overall_score: 82.4,
      dimensions: [
        { name: "Technical Skills", score: 88.5, weight: 0.35, weighted_score: 31.0, explanation: "Matched 4/5 required skills (keyword: 80%, semantic: 92%). Matched: machine learning, python, sql, tensorflow. Missing: aws." },
        { name: "Experience", score: 85.0, weight: 0.25, weighted_score: 21.3, explanation: "8 years detected (required: 5+). Log-scaled score: 85%. Significantly exceeds requirement." },
        { name: "Education", score: 100.0, weight: 0.15, weighted_score: 15.0, explanation: "Candidate: Masters (level 4), Required: Masters (level 4). Meets requirement." },
        { name: "Availability", score: 80.0, weight: 0.10, weighted_score: 8.0, explanation: "Notice period: 30 days (Within 1 month). Within acceptable range." },
        { name: "Miscellaneous", score: 71.0, weight: 0.15, weighted_score: 10.7, explanation: "Title match: 82% (best: 'Senior Data Scientist'). Overall relevance: 60%." }
      ],
      matched_skills: ["machine learning", "python", "sql", "tensorflow"],
      missing_skills: ["aws"], experience_years: 8, education_level: "Masters",
      job_titles: ["Senior Data Scientist", "Data Scientist"], notice_period_days: 30,
      justification: "Priya Sharma scores 82.4/100 overall. Key strengths: Technical Skills (89%), Experience (85%), Education (100%). Brings 8 years of industry experience."
    },
    {
      rank: 2, name: "Alex Chen", email: "alex.c@email.com", overall_score: 68.7,
      dimensions: [
        { name: "Technical Skills", score: 72.0, weight: 0.35, weighted_score: 25.2, explanation: "Matched 3/5 required skills." },
        { name: "Experience", score: 70.0, weight: 0.25, weighted_score: 17.5, explanation: "5 years detected. Meets requirement." },
        { name: "Education", score: 75.0, weight: 0.15, weighted_score: 11.3, explanation: "Candidate: Bachelors. Below Masters requirement." },
        { name: "Availability", score: 60.0, weight: 0.10, weighted_score: 6.0, explanation: "Notice period: 60 days." },
        { name: "Miscellaneous", score: 58.0, weight: 0.15, weighted_score: 8.7, explanation: "Title match: 65%." }
      ],
      matched_skills: ["python", "sql", "machine learning"], missing_skills: ["tensorflow", "aws"],
      experience_years: 5, education_level: "Bachelors",
      job_titles: ["Data Analyst", "ML Engineer"], notice_period_days: 60,
      justification: "Alex Chen scores 68.7/100. Solid technical foundation but education below requirement."
    },
    {
      rank: 3, name: "Jordan Lee", email: "jordan@email.com", overall_score: 51.2,
      dimensions: [
        { name: "Technical Skills", score: 45.0, weight: 0.35, weighted_score: 15.8, explanation: "Matched 2/5 required skills." },
        { name: "Experience", score: 40.0, weight: 0.25, weighted_score: 10.0, explanation: "2 years detected. Below minimum." },
        { name: "Education", score: 75.0, weight: 0.15, weighted_score: 11.3, explanation: "Candidate: Bachelors." },
        { name: "Availability", score: 90.0, weight: 0.10, weighted_score: 9.0, explanation: "Notice period: 14 days." },
        { name: "Miscellaneous", score: 34.0, weight: 0.15, weighted_score: 5.1, explanation: "Title match: 38%." }
      ],
      matched_skills: ["python", "sql"], missing_skills: ["machine learning", "tensorflow", "aws"],
      experience_years: 2, education_level: "Bachelors",
      job_titles: ["Junior Developer"], notice_period_days: 14,
      justification: "Jordan Lee scores 51.2/100. Areas of concern: Technical Skills (45%), Miscellaneous (34%)."
    }
  ]
};

function RadarChart({ dimensions, size = 200 }) {
  const cx = size / 2, cy = size / 2, r = size / 2 - 30;
  const n = dimensions.length;
  const angleStep = (2 * Math.PI) / n;

  const getPoint = (i, value) => {
    const angle = i * angleStep - Math.PI / 2;
    const dist = (value / 100) * r;
    return { x: cx + dist * Math.cos(angle), y: cy + dist * Math.sin(angle) };
  };

  const gridLevels = [0.25, 0.5, 0.75, 1.0];

  const dataPoints = dimensions.map((d, i) => getPoint(i, d.score));
  const pathData = dataPoints.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ") + "Z";

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {gridLevels.map((level) => {
        const pts = Array.from({ length: n }, (_, i) => getPoint(i, level * 100));
        const d = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ") + "Z";
        return <path key={level} d={d} fill="none" stroke="#ddd" strokeWidth="0.5" />;
      })}
      {dimensions.map((_, i) => {
        const end = getPoint(i, 100);
        return <line key={i} x1={cx} y1={cy} x2={end.x} y2={end.y} stroke="#e0e0e0" strokeWidth="0.5" />;
      })}
      <path d={pathData} fill={ORANGE + "30"} stroke={ORANGE} strokeWidth="2" />
      {dataPoints.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r="3.5" fill={ORANGE} />
      ))}
      {dimensions.map((d, i) => {
        const labelPoint = getPoint(i, 115);
        return (
          <text key={i} x={labelPoint.x} y={labelPoint.y} textAnchor="middle"
            dominantBaseline="central" fontSize="9" fill="#666" fontWeight="500">
            {d.name.replace("Technical Skills", "Tech").replace("Miscellaneous", "Misc")}
          </text>
        );
      })}
    </svg>
  );
}

function ScoreBar({ score, label, color = ORANGE }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 3, color: "#555" }}>
        <span>{label}</span>
        <span style={{ fontWeight: 600, color }}>{score.toFixed(1)}%</span>
      </div>
      <div style={{ height: 6, background: "#eee", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${Math.min(score, 100)}%`, height: "100%", background: color,
          borderRadius: 3, transition: "width 0.8s cubic-bezier(0.4, 0, 0.2, 1)" }} />
      </div>
    </div>
  );
}

function CandidateCard({ candidate, expanded, onToggle }) {
  const medalColors = { 1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32" };
  const medal = medalColors[candidate.rank];
  const scoreColor = candidate.overall_score >= 75 ? "#16a34a" :
    candidate.overall_score >= 50 ? ORANGE : "#dc2626";

  return (
    <div style={{
      background: CARD_BG, borderRadius: 12, padding: 20, marginBottom: 14,
      border: candidate.rank === 1 ? `2px solid ${ORANGE}` : "1px solid #e5e2dd",
      boxShadow: candidate.rank === 1 ? `0 4px 20px ${ORANGE}20` : "0 1px 4px rgba(0,0,0,0.04)",
      transition: "all 0.3s", cursor: "pointer",
    }} onClick={onToggle}>
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        <div style={{
          width: 44, height: 44, borderRadius: "50%",
          background: medal || NAVY, display: "flex", alignItems: "center",
          justifyContent: "center", color: "white", fontWeight: 700, fontSize: 18,
          fontFamily: "'Georgia', serif",
        }}>
          {candidate.rank}
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 700, fontSize: 17, color: NAVY, fontFamily: "'Georgia', serif" }}>
            {candidate.name}
          </div>
          <div style={{ fontSize: 12, color: "#888", marginTop: 2 }}>
            {candidate.job_titles?.[0] || "—"} &middot; {candidate.experience_years}y exp &middot; {candidate.education_level}
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: 28, fontWeight: 800, color: scoreColor, fontFamily: "'Georgia', serif" }}>
            {candidate.overall_score}
          </div>
          <div style={{ fontSize: 10, color: "#999", textTransform: "uppercase", letterSpacing: 1 }}>/100</div>
        </div>
      </div>

      {!expanded && (
        <div style={{ display: "flex", gap: 6, marginTop: 12, flexWrap: "wrap" }}>
          {candidate.dimensions.map((d) => (
            <span key={d.name} style={{
              fontSize: 11, padding: "3px 10px", borderRadius: 20,
              background: d.score >= 75 ? "#dcfce7" : d.score >= 50 ? "#fef3c7" : "#fee2e2",
              color: d.score >= 75 ? "#166534" : d.score >= 50 ? "#92400e" : "#991b1b",
              fontWeight: 500,
            }}>
              {d.name.replace("Technical Skills", "Tech").replace("Miscellaneous", "Misc")}: {d.score.toFixed(0)}%
            </span>
          ))}
        </div>
      )}

      {expanded && (
        <div style={{ marginTop: 16, animation: "fadeIn 0.3s ease" }}>
          <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
            <div style={{ flex: "1 1 280px" }}>
              {candidate.dimensions.map((d) => (
                <ScoreBar key={d.name} label={d.name} score={d.score}
                  color={d.score >= 75 ? "#16a34a" : d.score >= 50 ? ORANGE : "#dc2626"} />
              ))}
            </div>
            <div style={{ flex: "0 0 180px", display: "flex", justifyContent: "center" }}>
              <RadarChart dimensions={candidate.dimensions} size={180} />
            </div>
          </div>

          <div style={{ marginTop: 14, padding: 14, background: "#f8f7f4", borderRadius: 8, borderLeft: `3px solid ${ORANGE}` }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: NAVY, textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 6 }}>
              AI Justification
            </div>
            <div style={{ fontSize: 13, color: "#444", lineHeight: 1.6 }}>{candidate.justification}</div>
          </div>

          <div style={{ display: "flex", gap: 14, marginTop: 12, flexWrap: "wrap" }}>
            <div style={{ flex: 1, minWidth: 180 }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: NAVY, marginBottom: 6 }}>Matched Skills</div>
              <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                {candidate.matched_skills.map((s) => (
                  <span key={s} style={{ fontSize: 11, padding: "2px 8px", borderRadius: 4, background: "#dcfce7", color: "#166534" }}>{s}</span>
                ))}
              </div>
            </div>
            <div style={{ flex: 1, minWidth: 180 }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: "#991b1b", marginBottom: 6 }}>Missing Skills</div>
              <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                {candidate.missing_skills.map((s) => (
                  <span key={s} style={{ fontSize: 11, padding: "2px 8px", borderRadius: 4, background: "#fee2e2", color: "#991b1b" }}>{s}</span>
                ))}
              </div>
            </div>
          </div>

          {candidate.dimensions.map((d) => (
            <div key={d.name} style={{ marginTop: 8, fontSize: 12, color: "#666", lineHeight: 1.5 }}>
              <span style={{ fontWeight: 600, color: NAVY }}>{d.name}:</span> {d.explanation}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [jobTitle, setJobTitle] = useState("Senior Data Scientist");
  const [jobDesc, setJobDesc] = useState(
`We are looking for a Senior Data Scientist with 5+ years of experience in machine learning and data science.

Required Skills:
- Python, TensorFlow, PyTorch
- SQL, data pipeline design
- AWS cloud services
- Machine Learning model development and deployment
- Statistics and A/B testing

Education: Master's degree in Computer Science, Statistics, or related field.
Notice Period: Within 60 days preferred.`
  );
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [expandedIdx, setExpandedIdx] = useState(null);
  const [mode, setMode] = useState("input"); // "input" | "results"

  const handleRank = useCallback(async () => {
    setLoading(true);
    // Simulate API call with mock data
    // In production, replace with actual fetch to FastAPI backend:
    // const formData = new FormData();
    // formData.append('job_title', jobTitle);
    // formData.append('job_description', jobDesc);
    // files.forEach(f => formData.append('resumes', f));
    // const res = await fetch('http://localhost:8000/api/rank', { method: 'POST', body: formData });
    // const data = await res.json();
    await new Promise(r => setTimeout(r, 1500));
    setResults(MOCK_RESULT);
    setMode("results");
    setExpandedIdx(0);
    setLoading(false);
  }, [jobTitle, jobDesc]);

  return (
    <div style={{ minHeight: "100vh", background: LIGHT_BG, fontFamily: "'Segoe UI', system-ui, sans-serif" }}>
      <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        textarea:focus, input:focus { outline: none; border-color: ${ORANGE} !important; box-shadow: 0 0 0 3px ${ORANGE}20; }
      `}</style>

      {/* Header */}
      <div style={{ background: NAVY, padding: "16px 24px", display: "flex", alignItems: "center", gap: 16 }}>
        <div style={{
          width: 40, height: 40, borderRadius: 8, background: ORANGE,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontWeight: 900, color: "white", fontSize: 16, fontFamily: "'Georgia', serif",
        }}>SR</div>
        <div>
          <div style={{ color: "white", fontWeight: 700, fontSize: 18, fontFamily: "'Georgia', serif", letterSpacing: -0.3 }}>
            S-Rank ICRS
          </div>
          <div style={{ color: "#8ea8cc", fontSize: 11 }}>Intelligent Candidate Ranking System &middot; NUS-ISS MTech AI</div>
        </div>
        {results && (
          <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
            <button onClick={() => setMode("input")} style={{
              padding: "6px 16px", borderRadius: 6, border: "1px solid #ffffff40",
              background: mode === "input" ? ORANGE : "transparent",
              color: "white", fontSize: 12, cursor: "pointer", fontWeight: 600,
            }}>Job Description</button>
            <button onClick={() => setMode("results")} style={{
              padding: "6px 16px", borderRadius: 6, border: "1px solid #ffffff40",
              background: mode === "results" ? ORANGE : "transparent",
              color: "white", fontSize: 12, cursor: "pointer", fontWeight: 600,
            }}>Rankings</button>
          </div>
        )}
      </div>

      <div style={{ maxWidth: 900, margin: "0 auto", padding: 20 }}>
        {mode === "input" && (
          <div style={{ animation: "fadeIn 0.4s ease" }}>
            <div style={{ marginBottom: 20 }}>
              <label style={{ fontSize: 13, fontWeight: 700, color: NAVY, display: "block", marginBottom: 6 }}>
                Job Title
              </label>
              <input value={jobTitle} onChange={e => setJobTitle(e.target.value)}
                style={{
                  width: "100%", padding: "10px 14px", borderRadius: 8, border: "1px solid #ddd",
                  fontSize: 14, background: "white", boxSizing: "border-box",
                }} />
            </div>

            <div style={{ marginBottom: 20 }}>
              <label style={{ fontSize: 13, fontWeight: 700, color: NAVY, display: "block", marginBottom: 6 }}>
                Job Description
              </label>
              <textarea value={jobDesc} onChange={e => setJobDesc(e.target.value)}
                rows={12} style={{
                  width: "100%", padding: "12px 14px", borderRadius: 8, border: "1px solid #ddd",
                  fontSize: 13, lineHeight: 1.6, resize: "vertical", fontFamily: "inherit",
                  background: "white", boxSizing: "border-box",
                }} />
            </div>

            <div style={{
              padding: 20, background: "white", borderRadius: 10, border: "2px dashed #ccc",
              textAlign: "center", marginBottom: 20, cursor: "pointer",
            }}>
              <div style={{ fontSize: 28, marginBottom: 6 }}>+</div>
              <div style={{ fontSize: 14, fontWeight: 600, color: NAVY }}>Upload Resumes (PDF / DOCX)</div>
              <div style={{ fontSize: 12, color: "#999", marginTop: 4 }}>
                Drag and drop or click to select multiple files
              </div>
              <div style={{ fontSize: 11, color: ORANGE, marginTop: 8, fontWeight: 500 }}>
                Demo mode: Click "Rank Candidates" to see sample results
              </div>
            </div>

            {/* Dimension Weights */}
            <div style={{ background: "white", borderRadius: 10, padding: 16, border: "1px solid #e5e2dd", marginBottom: 20 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: NAVY, marginBottom: 12 }}>
                Scoring Weights (configurable — GA-optimized in production)
              </div>
              {[
                { key: "D1", label: "Technical Skills", w: 35, desc: "SBERT cosine + keyword overlap" },
                { key: "D2", label: "Experience", w: 25, desc: "Non-linear log scaling" },
                { key: "D3", label: "Education", w: 15, desc: "Ordinal comparison" },
                { key: "D4", label: "Availability", w: 10, desc: "Notice period tiers" },
                { key: "D5", label: "Miscellaneous", w: 15, desc: "Job-title SBERT similarity" },
              ].map(d => (
                <div key={d.key} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                  <span style={{ fontSize: 10, fontWeight: 700, color: ORANGE, width: 24 }}>{d.key}</span>
                  <span style={{ fontSize: 12, fontWeight: 600, color: NAVY, width: 120 }}>{d.label}</span>
                  <div style={{ flex: 1, height: 4, background: "#eee", borderRadius: 2, overflow: "hidden" }}>
                    <div style={{ width: `${d.w}%`, height: "100%", background: ORANGE, borderRadius: 2 }} />
                  </div>
                  <span style={{ fontSize: 12, fontWeight: 700, color: NAVY, width: 36, textAlign: "right" }}>{d.w}%</span>
                  <span style={{ fontSize: 10, color: "#999", width: 180 }}>{d.desc}</span>
                </div>
              ))}
            </div>

            <button onClick={handleRank} disabled={loading}
              style={{
                width: "100%", padding: "14px", borderRadius: 10, border: "none",
                background: loading ? "#999" : ORANGE, color: "white", fontSize: 16,
                fontWeight: 700, cursor: loading ? "default" : "pointer",
                fontFamily: "'Georgia', serif", letterSpacing: 0.3,
                transition: "all 0.2s",
              }}>
              {loading ? (
                <span style={{ animation: "pulse 1.2s infinite" }}>Analyzing resumes with S-Rank pipeline...</span>
              ) : (
                "Rank Candidates"
              )}
            </button>
          </div>
        )}

        {mode === "results" && results && (
          <div style={{ animation: "fadeIn 0.4s ease" }}>
            {/* Summary bar */}
            <div style={{
              display: "flex", gap: 14, marginBottom: 20, flexWrap: "wrap",
            }}>
              <div style={{ flex: 1, minWidth: 160, background: "white", borderRadius: 10, padding: 16, border: "1px solid #e5e2dd" }}>
                <div style={{ fontSize: 11, color: "#999", textTransform: "uppercase", letterSpacing: 0.5 }}>Role</div>
                <div style={{ fontSize: 16, fontWeight: 700, color: NAVY, fontFamily: "'Georgia', serif", marginTop: 4 }}>
                  {results.job_description.title}
                </div>
              </div>
              <div style={{ flex: "0 0 120px", background: "white", borderRadius: 10, padding: 16, border: "1px solid #e5e2dd", textAlign: "center" }}>
                <div style={{ fontSize: 11, color: "#999", textTransform: "uppercase", letterSpacing: 0.5 }}>Candidates</div>
                <div style={{ fontSize: 28, fontWeight: 800, color: NAVY, fontFamily: "'Georgia', serif", marginTop: 4 }}>
                  {results.total_candidates}
                </div>
              </div>
              <div style={{ flex: "0 0 120px", background: "white", borderRadius: 10, padding: 16, border: "1px solid #e5e2dd", textAlign: "center" }}>
                <div style={{ fontSize: 11, color: "#999", textTransform: "uppercase", letterSpacing: 0.5 }}>Top Score</div>
                <div style={{ fontSize: 28, fontWeight: 800, color: "#16a34a", fontFamily: "'Georgia', serif", marginTop: 4 }}>
                  {results.candidates[0]?.overall_score || 0}
                </div>
              </div>
            </div>

            {/* JD Skills */}
            <div style={{ marginBottom: 16, display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: NAVY }}>Required:</span>
              {results.job_description.required_skills.map(s => (
                <span key={s} style={{ fontSize: 11, padding: "2px 8px", borderRadius: 4, background: NAVY + "12", color: NAVY, fontWeight: 500 }}>{s}</span>
              ))}
            </div>

            {/* Candidate Cards */}
            {results.candidates.map((c, i) => (
              <CandidateCard key={i} candidate={c} expanded={expandedIdx === i}
                onToggle={() => setExpandedIdx(expandedIdx === i ? null : i)} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
