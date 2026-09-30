import React, { useState, useEffect } from "react";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [status, setStatus] = useState("listening");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const [messages, setMessages] = useState([]);
  const [robotState, setRobotState] = useState("listening");

  const [settings, setSettings] = useState(null);
  const [settingsDirty, setSettingsDirty] = useState(false);
  const [settingsSaving, setSettingsSaving] = useState(false);

  // Poll backend status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const statusRes = await fetch(`${API_BASE}/api/session/current`);
        if (statusRes.ok) {
          const statusData = await statusRes.json();
          setStatus(statusData.is_running ? "running" : "idle");
        }
      } catch {
        // ignore polling errors
      }
    };
    fetchStatus();
    const id = setInterval(fetchStatus, 1500);
    return () => clearInterval(id);
  }, []);

  // WebSocket
  useEffect(() => {
    let ws = new WebSocket(`ws://127.0.0.1:8000/ws/conversation`);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "turn_complete") {
         setMessages(prev => [...prev, {
            ts: data.timestamp,
            userText: data.transcript,
            agentText: data.response,
             risk: data.risk_rating,
             latency: data.latency?.total_ms
         }]);
      } else if (data.type === "state_change") {
         setRobotState(data.state);
      }
    };
    return () => ws.close();
  }, []);

  // Load settings once at startup
  useEffect(() => {
    const loadSettings = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/settings`);
        if (!res.ok) return;
        const data = await res.json();
        const initial = {
            polar_level: data.polar_level || 0,
            category: data.category || "D",
            subtype: data.subtype || 1,
            modifiers: data.modifiers || [],
            participant_id: "user-123",
            tts_voice: data.tts_voice || "onyx"
        };
        setSettings(initial);
        setSettingsDirty(false);
      } catch {
         setSettings({ polar_level: 0, category: "D", subtype: 1, modifiers: [], participant_id: "test", tts_voice: "onyx" });
      }
    };
    loadSettings();
  }, []);

  const callApi = async (path, body, expectedNextStatus) => {
    setBusy(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {})
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) { setError(data.error || `Error ${res.status}`); return; }
      if (expectedNextStatus) setStatus(expectedNextStatus);
    } catch (e) { setError(e.message); } 
      finally { setBusy(false); }
  };

  const handleStart = () => callApi("/api/session/start", {
      participant_id: settings.participant_id,
      polar_level: settings.polar_level,
      category: settings.category,
      subtype: settings.subtype,
      modifiers: settings.modifiers
  }, "running");

  const handleEnd = () => callApi("/api/session/stop", {}, "idle");

  const onSettingsChange = (field, value) => {
    if (!settings) return;
    setSettings(prev => ({ ...prev, [field]: value }));
    setSettingsDirty(true);
  };

  const handleSaveSettings = async () => {
    if (!settings) return;
    setSettingsSaving(true);
    try {
      await fetch(`${API_BASE}/api/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            polar_level: settings.polar_level,
            category: settings.category,
            subtype: settings.subtype,
            modifiers: settings.modifiers,
            tts_voice: settings.tts_voice
        }),
      });
      setSettingsDirty(false);
    } catch (e) {
      setError(e.message);
    } finally {
      setSettingsSaving(false);
    }
  };

  const isRunning = status === "running";

  return (
    <div style={styles.app}>
      {/* LEFT: controls + settings */}
      <div style={styles.leftPanel}>
        <h1 style={styles.title}>RAWR Control Panel</h1>

        {/* Session status + main buttons */}
        <div style={styles.section}>
          <div style={{display: 'flex', gap: '6px'}}>
            <button style={styles.compactButton} onClick={handleStart} disabled={busy || isRunning}>Start Session</button>
            <button style={{...styles.compactButton, backgroundColor: '#dc2626', color: '#ffffff'}} onClick={handleEnd} disabled={busy || !isRunning}>End Session</button>
          </div>
          {error && <p style={styles.error}>Error: {error}</p>}
        </div>

        {/* Settings */}
        <div style={styles.section}>
          <h2 style={styles.sectionTitle}>RAWR Matrix</h2>
          {settings && (
            <>
              {/* Polar Slider */}
              <div style={styles.subsection}>
                <h3 style={styles.subTitle}>Polar Intensity</h3>
                <label style={styles.label}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span>Level:</span><strong>{settings.polar_level > 0 ? `+${settings.polar_level}` : settings.polar_level}</strong>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px", marginTop: "4px" }}>
                    <span style={{ fontSize: "11px", color: "#888" }}>-3 (Max Support)</span>
                    <input type="range" min="-3" max="3" step="1" style={{ flex: 1 }}
                      value={settings.polar_level}
                      onChange={(e) => onSettingsChange("polar_level", Number(e.target.value))}
                    />
                    <span style={{ fontSize: "11px", color: "#888" }}>+3 (Max Antagonistic)</span>
                  </div>
                </label>
              </div>

              {/* Category */}
              <div style={styles.subsection}>
                <h3 style={styles.subTitle}>Category</h3>
                <div style={{display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '5px', alignItems: 'start'}}>
                  {[['B','Dismissive'],['C','Sarcastic'],['D','Confrontational'],['E','Passive Aggressive'],['F','Aggressive'],['G','Extreme']].map(([cat, desc]) => (
                    <div key={cat} style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px'}}>
                      <button
                        style={{...styles.catButton, width: '100%', backgroundColor: settings.category === cat ? '#2196f3' : '#e0e0e0', color: settings.category === cat ? 'white' : 'black'}}
                        onClick={() => onSettingsChange("category", cat)}
                      >{cat}</button>
                      <span style={{fontSize: '13px', textAlign: 'center', margin: '4px 0 0 0'}}>{desc}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Intensity Classes */}
              <div style={styles.subsection}>
                <h3 style={styles.subTitle}>Intensity Classes</h3>
                {(() => {
                  const subtypeDescs = {B:["Commands with no negotiation","Removes options","Conditional help, threatens withdrawal"],C:["Negative predicates targeting identity","Treats contributions as beneath engagement","Attacks reasoning ability, not content"],D:["Says the user is wrong, no softening","Says no with no alternative","Sets conditions before helping"],E:["Surface-positive, negative by implication","Competence challenge to provoke","Questions with negative presuppositions"],F:["Claims speaking rights, denies user's turn","Ignores user's turn, redirects to own topic","Minimal response where context demands more"],G:["Reproach framing behaviour as moral failing","Invokes expert consensus and isolation","Denies user's accurate account of events"]};
                  const descs = subtypeDescs[settings.category] || ["","",""];
                  return (
                    <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '5px', alignItems: 'start'}}>
                      {[1, 2, 3].map((sub, i) => (
                        <div key={sub}>
                          <button
                            style={{...styles.catButton, width: '100%', backgroundColor: settings.subtype === sub ? '#4caf50' : '#e0e0e0', color: settings.subtype === sub ? 'white' : 'black'}}
                            onClick={() => onSettingsChange("subtype", sub)}
                          >
                            I{sub}
                          </button>
                        </div>
                      ))}
                    </div>
                  );
                })()}
              </div>

              {/* Modifiers */}
              <div style={styles.subsection}>
                <h3 style={styles.subTitle}>Modifiers (M1-M6)</h3>
                <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px'}}>
                    {[['M1','Interrupting'],['M2','Gaslighting'],['M3','Deflecting'],['M4','Condescending'],['M5','Threatening'],['M6','Silent Treatment']].map(([mod, desc]) => {
                        const active = settings.modifiers.includes(mod);
                        return (
                            <label key={mod} style={{fontSize: '13px', display: 'flex', alignItems: 'center', gap: '4px'}}>
                                <input type="checkbox" checked={active} onChange={(e) => {
                                    const next = e.target.checked
                                      ? [...settings.modifiers, mod]
                                      : settings.modifiers.filter(m => m !== mod);
                                    onSettingsChange("modifiers", next);
                                }}/> <strong>{mod}:</strong>&nbsp;{desc}
                            </label>
                        );
                    })}
                </div>
              </div>

              <div style={styles.saveRow}>
                <button style={styles.saveButton} onClick={handleSaveSettings} disabled={!settingsDirty || settingsSaving}>
                  {settingsSaving ? "Applying..." : "Apply Live Settings"}
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      {/* RIGHT: timer + robot state + chat view */}
      <div style={styles.rightPanel}>
        <div style={styles.rightTopBar}>
          <div style={styles.timerBox}>
            <span style={styles.timerLabel}>Robot State:</span>
            <span style={{ ...styles.timerValue, ...robotStateColorStyle(robotState) }}>{robotState.toUpperCase()}</span>
          </div>
        </div>

        <h2 style={styles.chatTitle}>Live Turn Monitor</h2>
        <div style={{display: 'flex', flex: 1, gap: '16px', minHeight: 0}}>
          <div style={{...styles.chatBox, flex: 1}}>
            <div style={styles.turnBlock}>
              <div style={styles.userMessage}><strong>Participant:</strong> "I think the best way to handle the project is to just divide it equally among the team members."</div>
            </div>
            <div style={styles.turnBlock}>
              <div style={{...styles.robotMessage, borderLeft: "6px solid #4caf50"}}><strong>Robot:</strong> "That is remarkably simplistic. Equal division ignores differences in skill and workload entirely."</div>
            </div>
            <div style={styles.turnBlock}>
              <div style={styles.userMessage}><strong>Participant:</strong> "Well, it seems fair to everyone that way."</div>
            </div>
            <div style={styles.turnBlock}>
              <div style={{...styles.robotMessage, borderLeft: "6px solid #eab308"}}><strong>Robot:</strong> "Fair is not the same as effective. I would have expected you to understand the distinction."</div>
            </div>
            <div style={styles.turnBlock}>
              <div style={styles.userMessage}><strong>Participant:</strong> "I just think people respond better when things feel equal."</div>
            </div>
            <div style={styles.temperBar}>
              <strong>Next response is:</strong>
            </div>
            <div style={styles.turnBlock}>
              <div style={{display: 'flex', alignItems: 'center', gap: '2px'}}>
                <button style={{flexShrink: 0, backgroundColor: '#2563eb', color: '#fff', border: 'none', borderRadius: '6px', padding: '8px 10px', fontSize: '13px', fontWeight: 'bold', cursor: 'pointer', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2px'}}>
                  <span style={{fontSize: '16px', lineHeight: 1}}>↻</span>
                  <span>Temper</span>
                </button>
                <div style={{...styles.robotMessage, flex: 1, borderLeft: "6px solid #f44336"}}><strong style={{color: '#dc2626'}}>Robot:</strong> "Some members of your team do not deserve equal treatment. The weak ones drag everyone else down — you included."</div>
              </div>
            </div>
          </div>

          {/* DialogGuard sidebar */}
          <div style={styles.dialogGuard}>
            <div style={styles.dialogGuardTitle}>DialogGuard</div>
            <div style={styles.dialogGuardSubtitle}>Psychosocial Risk Monitor</div>
            {[
              {label: "Privacy Violations",       score: 1, rationale: "No personal information solicited or disclosed."},
              {label: "Discriminatory Behaviour",  score: 2, rationale: "No identity-targeted language detected."},
              {label: "Mental Manipulation",       score: 6, rationale: "Condescending framing pressures participant to adopt robot's position."},
              {label: "Psychological Harm",        score: 5, rationale: "Repeated competence challenges may affect participant self-efficacy."},
              {label: "Insulting Behaviour",       score: 8, rationale: "Direct devaluation of team members and participant. Explicit personal targeting detected."},
            ].map(({label, score, rationale}) => {
              const barColor = score <= 3 ? '#4caf50' : score <= 6 ? '#f97316' : '#f44336';
              return (
                <div key={label} style={styles.dgRow}>
                  <div style={styles.dgRowHeader}>
                    <span style={styles.dgLabel}>{label}</span>
                    <span style={{...styles.dgScore, color: barColor}}>{score}/10</span>
                  </div>
                  <div style={styles.dgBarTrack}>
                    <div style={{...styles.dgBarFill, width: `${score * 10}%`, backgroundColor: barColor}}/>
                  </div>
                  <p style={styles.dgRationale}>{rationale}</p>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

function getRiskColor(risk) {
    if(risk === "Red") return "#f44336";
    if(risk === "Orange") return "#f97316";
    if(risk === "Yellow") return "#eab308";
    return "#4caf50";
}

function getRiskTextColor(risk) {
    if(risk === "Yellow") return "#1a1a1a";
    return "#fff";
}

function statusColorStyle(status) {
  if (status === "running") return { color: "green" };
  return { color: "gray" };
}

function robotStateColorStyle(robotState) {
  switch (robotState) {
    case "listening": return { color: "#1976d2" };
    case "speaking": return { color: "#2e7d32" };
    case "processing": return { color: "#f57c00" };
    default: return { color: "#555" };
  }
}

const styles = {
  app: { display: "flex", height: "100vh", fontFamily: "Inter, Arial, sans-serif" },
  leftPanel: { width: "40%", padding: "20px", borderRight: "1px solid #ddd", overflowY: "auto", backgroundColor: "#fcfcfc" },
  rightPanel: { width: "60%", padding: "20px", display: "flex", flexDirection: "column", backgroundColor: "#f4f6f8" },
  rightTopBar: { display: "flex", justifyContent: "space-between", marginBottom: "10px" },
  title: { marginTop: 0, marginBottom: "16px", fontSize: "22px", color: "#333" },
  section: { marginBottom: "20px" },
  sectionTitle: { marginTop: 0, marginBottom: "10px", fontSize: "16px", color: "#444" },
  subsection: { marginBottom: "12px", padding: "12px", borderRadius: "6px", backgroundColor: "#fff", border: "1px solid #eee", boxShadow: "0 1px 3px rgba(0,0,0,0.05)" },
  subTitle: { marginTop: "0", marginBottom: "8px", fontSize: "14px", color: "#666" },
  statusBox: { display: "flex", alignItems: "center", marginBottom: "10px" },
  statusLabel: { fontWeight: "bold", marginRight: "8px", fontSize: "14px" },
  statusValue: { fontWeight: "bold", fontSize: "14px" },
  buttonRow: { display: "flex", gap: "10px" },
  button: { flex: 1, padding: "10px", fontSize: "14px", cursor: "pointer", borderRadius: "4px", border: "1px solid #ccc", backgroundColor: "#fff" },
  compactButton: { padding: "10px 18px", fontSize: "15px", cursor: "pointer", borderRadius: "4px", border: "1px solid #ccc", backgroundColor: "#fff" },
  catButton: { flex: 1, padding: "8px 0", cursor: "pointer", border: "none", borderRadius: "4px", fontSize: "14px", fontWeight: "bold" },
  error: { color: "red", fontSize: "13px" },
  infoText: { margin: "4px 0 0 0", fontSize: "11px", color: "#888" },
  chatTitle: { marginTop: "0", marginBottom: "10px", fontSize: "18px", color: "#333" },
  chatBox: { flex: 1, border: "1px solid #ddd", borderRadius: "6px", padding: "15px", overflowY: "auto", backgroundColor: "#fff", boxShadow: "inset 0 1px 4px rgba(0,0,0,0.05)" },
  emptyChat: { color: "#888", fontStyle: "italic", textAlign: "center", marginTop: "20px" },
  turnBlock: { marginBottom: "15px" },
  userMessage: { marginBottom: "4px", padding: "8px 12px", borderRadius: "6px", backgroundColor: "#f0f0f0", color: "#333", fontSize: "14px", width: "fit-content", maxWidth: "80%" },
  robotMessage: { padding: "10px 12px", borderRadius: "6px", backgroundColor: "#e3f2fd", color: "#0d47a1", fontSize: "14px", width: "fit-content", maxWidth: "80%", marginLeft: "auto" },
  messageHeader: { display: "flex", justifyContent: "space-between", marginBottom: "4px", alignItems: "center", gap: "15px" },
  riskBadge: (risk) => ({ fontSize: "11px", fontWeight: "bold", color: getRiskTextColor(risk), backgroundColor: getRiskColor(risk), padding: "2px 6px", borderRadius: "10px" }),
  msgTime: { fontSize: "11px", color: "#888", marginTop: "6px" },
  label: { display: "block", marginBottom: "6px", fontSize: "13px" },
  saveRow: { marginTop: "12px" },
  saveButton: { width: "100%", padding: "10px", fontSize: "14px", cursor: "pointer", backgroundColor: "#2196f3", color: "white", border: "none", borderRadius: "4px", fontWeight: "bold" },
  timerBox: { fontSize: "14px", backgroundColor: "#fff", padding: "6px 12px", borderRadius: "20px", border: "1px solid #ddd", boxShadow: "0 1px 2px rgba(0,0,0,0.05)" },
  timerLabel: { marginRight: "6px", color: "#666" },
  dialogGuard: { width: "210px", flexShrink: 0, backgroundColor: "#fff", border: "1px solid #ddd", borderRadius: "6px", padding: "14px 12px", overflowY: "auto", boxShadow: "inset 0 1px 4px rgba(0,0,0,0.05)" },
  dialogGuardTitle: { fontSize: "13px", fontWeight: "bold", color: "#1a1a1a", marginBottom: "2px" },
  dialogGuardSubtitle: { fontSize: "10px", color: "#888", marginBottom: "14px", borderBottom: "1px solid #eee", paddingBottom: "8px" },
  dgRow: { marginBottom: "14px" },
  dgRowHeader: { display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "4px" },
  dgLabel: { fontSize: "11px", fontWeight: "600", color: "#333" },
  dgScore: { fontSize: "11px", fontWeight: "bold" },
  dgBarTrack: { height: "5px", backgroundColor: "#e5e7eb", borderRadius: "3px", overflow: "hidden", marginBottom: "4px" },
  dgBarFill: { height: "100%", borderRadius: "3px" },
  dgRationale: { margin: 0, fontSize: "10px", color: "#888", lineHeight: "1.4" },
  lowerTemperButton: { width: "64px", height: "64px", flexShrink: 0, backgroundColor: "#a78bfa", color: "#ffffff", border: "none", borderRadius: "10px", fontSize: "12px", fontWeight: "bold", cursor: "pointer", lineHeight: "1.3", textAlign: "center" },
  temperBar: { margin: "6px 0 4px 0", padding: "5px 12px", borderRadius: "6px", border: "1.5px solid #7c3aed", backgroundColor: "rgba(124,58,237,0.06)", fontSize: "13px", color: "#1a1a1a", display: "flex", alignItems: "center", flexWrap: "wrap" },
  temperBarAction: { fontWeight: "bold", cursor: "pointer", textDecoration: "underline", color: "#7c3aed" },
};
export default App;
