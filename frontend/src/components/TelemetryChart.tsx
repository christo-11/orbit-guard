import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Satellite } from "../types/satellite";

interface TelemetryChartProps {
  satellite: Satellite;
}

const signals = [
  {
    key: "batteryVoltage",
    label: "Battery voltage",
    color: "#00F0FF",
    unit: "V",
    digits: 2,
  },
  {
    key: "solarPanelTemp",
    label: "Solar panel temp",
    color: "#FFB800",
    unit: "°C",
    digits: 1,
  },
  {
    key: "gyroDrift",
    label: "Gyro drift",
    color: "#A78BFA",
    unit: "°/h",
    digits: 4,
  },
] as const;

export function TelemetryChart({ satellite }: TelemetryChartProps) {
  const chartData = satellite.telemetry24h.timestamps.map((time, index) => ({
    time,
    batteryVoltage: satellite.telemetry24h.batteryVoltage[index],
    solarPanelTemp: satellite.telemetry24h.solarPanelTemp[index],
    gyroDrift: satellite.telemetry24h.gyroDrift[index],
  }));

  return (
    <section className="glass-card animate-fade-up p-5 sm:p-6">
      <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="eyebrow">Signal history</p>
          <h2 className="mt-1 text-lg font-semibold text-white">
            24-hour telemetry
          </h2>
        </div>
        <span className="rounded-full border border-slate-700/70 bg-slate-950/40 px-3 py-1.5 font-mono text-[11px] text-slate-400">
          {satellite.id}
        </span>
      </div>

      <div className="mb-5 flex flex-wrap gap-x-5 gap-y-2">
        {signals.map((signal) => (
          <div className="flex items-center gap-2 text-xs text-slate-400" key={signal.key}>
            <span
              aria-hidden="true"
              className="h-2 w-2 rounded-full"
              style={{ backgroundColor: signal.color, boxShadow: `0 0 12px ${signal.color}66` }}
            />
            {signal.label}
          </div>
        ))}
      </div>

      <div className="h-[270px] w-full" role="img" aria-label={`24-hour battery, solar temperature, and gyro telemetry for ${satellite.name}`}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
            <CartesianGrid stroke="rgba(148,163,184,.1)" strokeDasharray="3 5" />
            <XAxis
              axisLine={false}
              dataKey="time"
              tick={{ fill: "#64748b", fontSize: 10 }}
              tickLine={false}
              tickMargin={10}
              interval={5}
            />
            <YAxis
              axisLine={false}
              tick={{ fill: "#64748b", fontSize: 10 }}
              tickLine={false}
              tickCount={5}
              width={42}
            />
            <Tooltip
              contentStyle={{
                background: "#0f172a",
                border: "1px solid rgba(148,163,184,.22)",
                borderRadius: 12,
                color: "#e2e8f0",
                fontSize: 12,
              }}
              formatter={(value, name) => {
                const signal = signals.find((item) => item.key === name);
                return [
                  `${Number(value).toFixed(signal?.digits ?? 2)} ${signal?.unit ?? ""}`,
                  signal?.label ?? name,
                ];
              }}
              labelStyle={{ color: "#94a3b8", marginBottom: 4 }}
            />
            {signals.map((signal) => (
              <Line
                activeDot={{ r: 4, strokeWidth: 0 }}
                dataKey={signal.key}
                dot={false}
                key={signal.key}
                name={signal.label}
                stroke={signal.color}
                strokeWidth={2}
                type="monotone"
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-2 text-[11px] text-slate-500">
        Simulated sample telemetry · hourly values · independent signal scales
      </p>
    </section>
  );
}
