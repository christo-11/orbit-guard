import type {
  HealthStatus,
  Satellite,
  Subsystem,
  Telemetry24h,
} from "../types/satellite";

interface SatelliteSeed {
  id: string;
  name: string;
  altitudeKm: number;
  inclinationDeg: number;
  speedKmh: number;
  healthStatus: HealthStatus;
  anomalyProbability: number;
  affectedSubsystem: Subsystem;
  aiExplanation: string;
  batteryBase: number;
  batteryTrend: number;
  temperatureBase: number;
  temperatureSwing: number;
  gyroBase: number;
  gyroTrend: number;
  phase: number;
}

function createTelemetry(seed: SatelliteSeed): Telemetry24h {
  const timestamps = Array.from({ length: 24 }, (_, hour) =>
    `${String(hour).padStart(2, "0")}:00`,
  );
  const batteryVoltage = timestamps.map((_, hour) =>
    Number(
      (
        seed.batteryBase +
        seed.batteryTrend * (hour / 23) +
        0.18 * Math.sin((hour + seed.phase) * (Math.PI / 6))
      ).toFixed(2),
    ),
  );
  const solarPanelTemp = timestamps.map((_, hour) =>
    Number(
      (
        seed.temperatureBase +
        seed.temperatureSwing * Math.sin((hour + seed.phase) * (Math.PI / 12))
      ).toFixed(1),
    ),
  );
  const gyroDrift = timestamps.map((_, hour) =>
    Number(
      (
        seed.gyroBase +
        seed.gyroTrend * (hour / 23) +
        0.003 * Math.sin((hour + seed.phase) * (Math.PI / 8))
      ).toFixed(4),
    ),
  );

  return { timestamps, batteryVoltage, solarPanelTemp, gyroDrift };
}

const satelliteSeeds: SatelliteSeed[] = [
  {
    id: "SAT-001",
    name: "Asteria-1",
    altitudeKm: 547,
    inclinationDeg: 97.6,
    speedKmh: 27_420,
    healthStatus: "Nominal",
    anomalyProbability: 4.2,
    affectedSubsystem: "Power",
    aiExplanation:
      "All monitored signals are within their usual operating ranges. Battery voltage and attitude stability are holding steady.",
    batteryBase: 30.1,
    batteryTrend: 0.08,
    temperatureBase: 12,
    temperatureSwing: 25,
    gyroBase: 0.008,
    gyroTrend: 0.002,
    phase: 1,
  },
  {
    id: "SAT-002",
    name: "Kepler Scout",
    altitudeKm: 612,
    inclinationDeg: 98.2,
    speedKmh: 27_180,
    healthStatus: "Warning",
    anomalyProbability: 68.4,
    affectedSubsystem: "Thermal",
    aiExplanation:
      "Solar panel temperatures are running warmer than this satellite's recent baseline. The trend is persistent, so thermal telemetry should be reviewed.",
    batteryBase: 29.4,
    batteryTrend: -0.62,
    temperatureBase: 27,
    temperatureSwing: 32,
    gyroBase: 0.014,
    gyroTrend: 0.009,
    phase: 4,
  },
  {
    id: "SAT-003",
    name: "Polaris Relay",
    altitudeKm: 698,
    inclinationDeg: 88.0,
    speedKmh: 26_880,
    healthStatus: "Critical",
    anomalyProbability: 94.7,
    affectedSubsystem: "Attitude",
    aiExplanation:
      "Gyroscope drift has increased throughout the last day and is well above its normal range. Confirm attitude-control telemetry and schedule an operator review.",
    batteryBase: 28.9,
    batteryTrend: -0.35,
    temperatureBase: 8,
    temperatureSwing: 28,
    gyroBase: 0.025,
    gyroTrend: 0.082,
    phase: 7,
  },
  {
    id: "SAT-004",
    name: "Lumen Mapper",
    altitudeKm: 522,
    inclinationDeg: 97.4,
    speedKmh: 27_460,
    healthStatus: "Nominal",
    anomalyProbability: 7.8,
    affectedSubsystem: "Communications",
    aiExplanation:
      "Telemetry signals are stable and no strong anomaly pattern is present. The communications subsystem is responding normally.",
    batteryBase: 30.4,
    batteryTrend: 0.03,
    temperatureBase: 5,
    temperatureSwing: 24,
    gyroBase: 0.009,
    gyroTrend: 0.001,
    phase: 10,
  },
  {
    id: "SAT-005",
    name: "Vega Pathfinder",
    altitudeKm: 735,
    inclinationDeg: 99.1,
    speedKmh: 26_790,
    healthStatus: "Warning",
    anomalyProbability: 52.1,
    affectedSubsystem: "Power",
    aiExplanation:
      "Battery voltage has drifted down slightly over several orbital cycles. The change is not yet critical, but continued monitoring is recommended.",
    batteryBase: 29.8,
    batteryTrend: -0.92,
    temperatureBase: 15,
    temperatureSwing: 27,
    gyroBase: 0.011,
    gyroTrend: 0.006,
    phase: 14,
  },
  {
    id: "SAT-006",
    name: "Aurora Observer",
    altitudeKm: 581,
    inclinationDeg: 96.8,
    speedKmh: 27_310,
    healthStatus: "Nominal",
    anomalyProbability: 2.6,
    affectedSubsystem: "Thermal",
    aiExplanation:
      "No meaningful deviation detected. Solar panel temperature follows its expected day-night cycle and the other signals are stable.",
    batteryBase: 30.6,
    batteryTrend: 0.12,
    temperatureBase: 10,
    temperatureSwing: 23,
    gyroBase: 0.006,
    gyroTrend: 0.001,
    phase: 18,
  },
];

export const mockSatellites: Satellite[] = satelliteSeeds.map((seed) => ({
  id: seed.id,
  name: seed.name,
  orbit: {
    altitudeKm: seed.altitudeKm,
    inclinationDeg: seed.inclinationDeg,
    speedKmh: seed.speedKmh,
  },
  healthStatus: seed.healthStatus,
  anomalyProbability: seed.anomalyProbability,
  affectedSubsystem: seed.affectedSubsystem,
  aiExplanation: seed.aiExplanation,
  telemetry24h: createTelemetry(seed),
}));
