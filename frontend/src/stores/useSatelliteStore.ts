import { create } from "zustand";
import { mockSatellites } from "../data/mockSatellites";
import type { HealthStatus, Satellite } from "../types/satellite";

interface SatelliteStore {
  satellites: Satellite[];
  selectedSatelliteId: string;
  statusFilter: HealthStatus | "All";
  selectSatellite: (satelliteId: string) => void;
  setStatusFilter: (status: HealthStatus | "All") => void;
  getSelectedSatellite: () => Satellite;
}

export const useSatelliteStore = create<SatelliteStore>((set, get) => ({
  satellites: mockSatellites,
  selectedSatelliteId: mockSatellites[0].id,
  statusFilter: "All",
  selectSatellite: (satelliteId) => {
    if (get().satellites.some((satellite) => satellite.id === satelliteId)) {
      set({ selectedSatelliteId: satelliteId });
    }
  },
  setStatusFilter: (statusFilter) =>
    set((state) => {
      const visibleSatellites =
        statusFilter === "All"
          ? state.satellites
          : state.satellites.filter(
              (satellite) => satellite.healthStatus === statusFilter,
            );
      const selectionIsVisible = visibleSatellites.some(
        (satellite) => satellite.id === state.selectedSatelliteId,
      );

      return {
        statusFilter,
        selectedSatelliteId: selectionIsVisible
          ? state.selectedSatelliteId
          : (visibleSatellites[0]?.id ?? state.selectedSatelliteId),
      };
    }),
  getSelectedSatellite: () =>
    get().satellites.find(
      (satellite) => satellite.id === get().selectedSatelliteId,
    ) ?? get().satellites[0],
}));
