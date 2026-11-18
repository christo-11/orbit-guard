export type HealthStatus = "Nominal" | "Warning" | "Critical";

export type Subsystem =
  | "Thermal"
  | "Power"
  | "Attitude"
  | "Communications"
  | "Propulsion";

export interface OrbitParameters {
  altitudeKm: number;
  inclinationDeg: number;
  speedKmh: number;
}

export interface Telemetry24h {
  timestamps: string[];
  batteryVoltage: number[];
  solarPanelTemp: number[];
  gyroDrift: number[];
}

export interface Satellite {
  id: string;
  name: string;
  orbit: OrbitParameters;
  healthStatus: HealthStatus;
  anomalyProbability: number;
  affectedSubsystem: Subsystem;
  aiExplanation: string;
  telemetry24h: Telemetry24h;
}
