import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  Bell,
  ChevronDown,
  CircleHelp,
  Command,
  Crosshair,
  LayoutDashboard,
  Orbit,
  Radio,
  Satellite as SatelliteIcon,
  ShieldCheck,
  Signal,
  Sparkles,
} from "lucide-react";
import { TelemetryChart } from "./components/TelemetryChart";
import { useSatelliteStore } from "./stores/useSatelliteStore";
import type { HealthStatus, Satellite } from "./types/satellite";

const statusStyles: Record<HealthStatus, string> = {
  Nominal: "status-nominal",
  Warning: "status-warning",
  Critical: "status-critical",
};

const statusBarStyles: Record<HealthStatus, string> = {
  Nominal: "bg-mission-nominal",
  Warning: "bg-mission-warning",
  Critical: "bg-mission-critical",
};

function formatProbability(probability: number) {
  return `${probability.toFixed(1)}%`;
}

function SatelliteCard({
  satellite,
  selected,
  onSelect,
}: {
  satellite: Satellite;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      aria-pressed={selected}
      className={`satellite-card group text-left ${selected ? "satellite-card-selected" : ""}`}
      onClick={onSelect}
      type="button"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <span className="satellite-icon">
            <SatelliteIcon aria-hidden="true" size={18} />
          </span>
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-slate-100">{satellite.name}</p>
            <p className="mt-1 font-mono text-[10px] tracking-wide text-slate-500">{satellite.id}</p>
          </div>
        </div>
        <span className={`status-pill ${statusStyles[satellite.healthStatus]}`}>
          <span className="status-dot" />
          {satellite.healthStatus}
        </span>
      </div>

      <div className="mt-5 flex items-end justify-between gap-3">
        <div>
          <p className="text-[11px] text-slate-500">Anomaly probability</p>
          <p className="mt-1 font-mono text-2xl font-semibold tracking-tight text-white">
            {formatProbability(satellite.anomalyProbability)}
          </p>
        </div>
        <span className="mb-1 rounded-lg border border-slate-700/60 bg-slate-950/30 px-2.5 py-1.5 text-[10px] text-slate-400">
          {satellite.affectedSubsystem}
        </span>
      </div>

      <div className="mt-4 h-1 overflow-hidden rounded-full bg-slate-800">
        <div
          className={`h-full rounded-full transition-all duration-700 ${statusBarStyles[satellite.healthStatus]}`}
          style={{ width: `${satellite.anomalyProbability}%` }}
        />
      </div>
      <div className="mt-4 flex items-center justify-between border-t border-slate-800/80 pt-3 text-[10px] text-slate-500">
        <span>{satellite.orbit.altitudeKm} km altitude</span>
        <span className="inline-flex items-center gap-1 text-slate-400 transition-colors group-hover:text-mission-nominal">
          View telemetry <ArrowUpRight aria-hidden="true" size={12} />
        </span>
      </div>
    </button>
  );
}

function OrbitalVisual() {
  return (
    <div aria-label="Decorative animated orbital visualization" className="orbital-visual" role="img">
      <div className="orbital-grid" />
      <div className="orbit-ring orbit-ring-one">
        <span className="orbit-object orbit-object-one" />
      </div>
      <div className="orbit-ring orbit-ring-two">
        <span className="orbit-object orbit-object-two" />
      </div>
      <div className="orbit-ring orbit-ring-three">
        <span className="orbit-object orbit-object-three" />
      </div>
      <div className="planet">
        <div className="planet-highlight" />
        <div className="planet-grid-lines" />
      </div>
      <div className="orbital-caption">
        <span className="h-1.5 w-1.5 animate-soft-pulse rounded-full bg-mission-nominal" />
        ORBITAL TRACKS · SIMULATED
      </div>
    </div>
  );
}

function App() {
  const satellites = useSatelliteStore((state) => state.satellites);
  const selectedSatelliteId = useSatelliteStore((state) => state.selectedSatelliteId);
  const statusFilter = useSatelliteStore((state) => state.statusFilter);
  const selectSatellite = useSatelliteStore((state) => state.selectSatellite);
  const setStatusFilter = useSatelliteStore((state) => state.setStatusFilter);
  const selectedSatellite = useSatelliteStore((state) => state.getSelectedSatellite());

  const filteredSatellites =
    statusFilter === "All"
      ? satellites
      : satellites.filter((satellite) => satellite.healthStatus === statusFilter);
  const criticalCount = satellites.filter((satellite) => satellite.healthStatus === "Critical").length;
  const warningCount = satellites.filter((satellite) => satellite.healthStatus === "Warning").length;
  const nominalCount = satellites.filter((satellite) => satellite.healthStatus === "Nominal").length;
  const highRisk = satellites.reduce(
    (highest, satellite) =>
      satellite.anomalyProbability > highest.anomalyProbability ? satellite : highest,
    satellites[0],
  );

  return (
    <div className="min-h-screen bg-mission-background text-slate-100">
      <aside className="sidebar">
        <a className="brand-lockup" href="#" aria-label="ORBITGUARD mission control home">
          <span className="brand-mark">
            <Orbit aria-hidden="true" size={21} />
          </span>
          <span>
            <span className="block text-[13px] font-bold tracking-[.16em] text-white">ORBITGUARD</span>
            <span className="mt-1 block text-[9px] tracking-[.18em] text-slate-500">MISSION CONTROL</span>
          </span>
        </a>

        <div className="sidebar-label">WORKSPACE</div>
        <nav aria-label="Main navigation" className="space-y-1">
          <a className="nav-link nav-link-active" href="#overview">
            <LayoutDashboard aria-hidden="true" size={16} /> Overview
          </a>
          <a className="nav-link" href="#satellites">
            <SatelliteIcon aria-hidden="true" size={16} /> Satellites
            <span className="nav-count">{satellites.length}</span>
          </a>
          <a className="nav-link" href="#telemetry">
            <Activity aria-hidden="true" size={16} /> Telemetry
          </a>
          <a className="nav-link" href="#risk-review">
            <Crosshair aria-hidden="true" size={16} /> Risk review
            {criticalCount > 0 && <span className="nav-alert">{criticalCount}</span>}
          </a>
        </nav>

        <div className="sidebar-label mt-9">SYSTEM</div>
        <div className="system-card">
          <div className="flex items-center gap-2 text-xs font-medium text-slate-200">
            <Signal aria-hidden="true" className="text-mission-nominal" size={15} />
            Simulation ready
          </div>
          <p className="mt-2 text-[10px] leading-relaxed text-slate-500">
            Mock telemetry · local demo only
          </p>
          <div className="mt-3 flex items-center gap-2 font-mono text-[9px] text-slate-500">
            <span className="h-1.5 w-1.5 animate-soft-pulse rounded-full bg-mission-nominal" />
            DEMO FEED ACTIVE
          </div>
        </div>

        <a className="nav-link sidebar-help" href="#risk-review">
          <CircleHelp aria-hidden="true" size={16} /> Guide & definitions
        </a>
        <div className="sidebar-profile">
          <span className="profile-avatar">OC</span>
          <span className="min-w-0 flex-1">
            <span className="block truncate text-xs font-medium text-slate-200">Operations Console</span>
            <span className="mt-1 block text-[10px] text-slate-500">Flight systems</span>
          </span>
          <ChevronDown aria-hidden="true" className="text-slate-500" size={14} />
        </div>
      </aside>

      <main className="main-content" id="overview">
        <header className="topbar">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <span>Workspace</span>
            <span className="text-slate-700">/</span>
            <span className="text-slate-300">Fleet overview</span>
          </div>
          <div className="flex items-center gap-3 sm:gap-5">
            <span className="hidden items-center gap-2 font-mono text-[10px] text-slate-500 sm:flex">
              <span className="h-1.5 w-1.5 rounded-full bg-mission-nominal" />
              UTC {new Date().toISOString().slice(11, 16)}
            </span>
            <button aria-label="Notifications" className="icon-button" type="button">
              <Bell aria-hidden="true" size={16} />
              <span className="notification-dot" />
            </button>
            <span className="hidden items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-900/60 px-2.5 py-2 font-mono text-[10px] text-slate-400 md:flex">
              <Command aria-hidden="true" size={12} /> MISSION / 01
            </span>
          </div>
        </header>

        <div className="page-shell">
          <section className="hero-panel">
            <div className="hero-copy">
              <div className="eyebrow flex items-center gap-2">
                <span className="h-1.5 w-1.5 animate-soft-pulse rounded-full bg-mission-nominal" />
                FLEET OVERVIEW <span className="text-slate-700">/</span> MOCK DATA
              </div>
              <h1 className="mt-4 max-w-xl text-[clamp(2rem,4vw,3.15rem)] font-semibold leading-[1.08] tracking-[-.055em] text-white">
                A clearer view
                <br />
                <span className="text-gradient">of what’s in orbit.</span>
              </h1>
              <p className="mt-4 max-w-lg text-sm leading-7 text-slate-400">
                Monitor satellite health, spot emerging anomalies, and understand what the signals may mean.
              </p>
              <div className="mt-6 flex flex-wrap items-center gap-3">
                <a className="primary-link" href="#satellites">
                  <SatelliteIcon aria-hidden="true" size={15} />
                  Explore fleet
                  <ArrowUpRight aria-hidden="true" size={14} />
                </a>
                <span className="text-[11px] text-slate-500">6 assets tracked · 24h signal window</span>
              </div>
            </div>
            <OrbitalVisual />
            <div className="hero-bottom-line" />
          </section>

          <section aria-label="Fleet health summary" className="summary-grid">
            <div className="summary-card">
              <div className="summary-icon text-mission-nominal"><SatelliteIcon aria-hidden="true" size={17} /></div>
              <div className="mt-4 flex items-end justify-between gap-2">
                <div>
                  <p className="text-xs text-slate-500">Satellites tracked</p>
                  <p className="mt-1 font-mono text-2xl font-semibold text-white">{satellites.length}</p>
                </div>
                <span className="summary-change"><ArrowUpRight aria-hidden="true" size={12} /> Fleet online</span>
              </div>
            </div>
            <div className="summary-card">
              <div className="summary-icon text-mission-nominal"><ShieldCheck aria-hidden="true" size={17} /></div>
              <div className="mt-4 flex items-end justify-between gap-2">
                <div>
                  <p className="text-xs text-slate-500">Nominal</p>
                  <p className="mt-1 font-mono text-2xl font-semibold text-white">{nominalCount}</p>
                </div>
                <span className="summary-change">{Math.round((nominalCount / satellites.length) * 100)}% of fleet</span>
              </div>
            </div>
            <div className="summary-card">
              <div className="summary-icon text-mission-warning"><Activity aria-hidden="true" size={17} /></div>
              <div className="mt-4 flex items-end justify-between gap-2">
                <div>
                  <p className="text-xs text-slate-500">Needs attention</p>
                  <p className="mt-1 font-mono text-2xl font-semibold text-white">{warningCount + criticalCount}</p>
                </div>
                <span className="summary-change text-mission-warning">{warningCount} warnings · {criticalCount} critical</span>
              </div>
            </div>
            <div className="summary-card">
              <div className="summary-icon text-mission-critical"><Activity aria-hidden="true" size={17} /></div>
              <div className="mt-4 flex items-end justify-between gap-2">
                <div>
                  <p className="text-xs text-slate-500">Highest anomaly risk</p>
                  <p className="mt-1 font-mono text-2xl font-semibold text-white">{formatProbability(highRisk.anomalyProbability)}</p>
                </div>
                <span className="summary-change text-slate-400">{highRisk.id}</span>
              </div>
            </div>
          </section>

          <section className="mt-9" id="satellites">
            <div className="section-heading">
              <div>
                <p className="eyebrow">FLEET STATUS</p>
                <h2 className="mt-1.5 text-xl font-semibold tracking-tight text-white">Satellite overview</h2>
                <p className="mt-1 text-xs text-slate-500">Select an asset to inspect its telemetry and risk summary.</p>
              </div>
              <label className="filter-control">
                <span className="sr-only">Filter satellites by health</span>
                <select
                  aria-label="Filter satellites by health"
                  onChange={(event) => setStatusFilter(event.target.value as HealthStatus | "All")}
                  value={statusFilter}
                >
                  <option value="All">All health states</option>
                  <option value="Nominal">Nominal</option>
                  <option value="Warning">Warning</option>
                  <option value="Critical">Critical</option>
                </select>
                <ChevronDown aria-hidden="true" size={14} />
              </label>
            </div>

            <div className="satellite-grid">
              {filteredSatellites.map((satellite) => (
                <SatelliteCard
                  key={satellite.id}
                  onSelect={() => selectSatellite(satellite.id)}
                  satellite={satellite}
                  selected={satellite.id === selectedSatelliteId}
                />
              ))}
            </div>
          </section>

          <section className="detail-grid mt-8" id="telemetry">
            <TelemetryChart satellite={selectedSatellite} />
            <aside className="glass-card risk-card animate-fade-up" id="risk-review">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="eyebrow">AI RISK REVIEW</p>
                  <h2 className="mt-1 text-lg font-semibold text-white">{selectedSatellite.name}</h2>
                </div>
                <span className={`status-pill ${statusStyles[selectedSatellite.healthStatus]}`}>
                  <span className="status-dot" />
                  {selectedSatellite.healthStatus}
                </span>
              </div>

              <div className="mt-5 rounded-xl border border-slate-800/80 bg-slate-950/40 p-4">
                <div className="flex items-center gap-2 text-xs font-medium text-slate-300">
                  <Sparkles aria-hidden="true" className="text-mission-nominal" size={14} />
                  Plain-English assessment
                </div>
                <p className="mt-3 text-xs leading-6 text-slate-400">{selectedSatellite.aiExplanation}</p>
              </div>

              <div className="mt-5 space-y-4">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-500">Affected subsystem</span>
                  <span className="rounded-md border border-slate-700/70 bg-slate-950/40 px-2 py-1 text-slate-300">
                    {selectedSatellite.affectedSubsystem}
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-3 border-t border-slate-800/80 pt-4">
                  <div>
                    <p className="text-[10px] text-slate-500">Altitude</p>
                    <p className="mt-1 font-mono text-xs text-slate-200">{selectedSatellite.orbit.altitudeKm} km</p>
                  </div>
                  <div>
                    <p className="text-[10px] text-slate-500">Inclination</p>
                    <p className="mt-1 font-mono text-xs text-slate-200">{selectedSatellite.orbit.inclinationDeg}°</p>
                  </div>
                  <div>
                    <p className="text-[10px] text-slate-500">Orbital speed</p>
                    <p className="mt-1 font-mono text-xs text-slate-200">{selectedSatellite.orbit.speedKmh.toLocaleString()} km/h</p>
                  </div>
                </div>
              </div>
              <div className="mt-5 flex items-start gap-2 rounded-lg border border-slate-800/70 bg-slate-900/40 p-3 text-[10px] leading-5 text-slate-500">
                <Radio aria-hidden="true" className="mt-0.5 shrink-0 text-slate-400" size={13} />
                Mock prediction for interface demonstration. Not flight telemetry or operational guidance.
              </div>
            </aside>
          </section>

          <footer className="page-footer">
            <span>ORBITGUARD <span className="text-slate-700">·</span> MISSION INTELLIGENCE</span>
            <span className="inline-flex items-center gap-1.5"><ArrowDownRight aria-hidden="true" size={12} /> Mock data for demonstration</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

export default App;
