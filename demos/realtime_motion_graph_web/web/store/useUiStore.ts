"use client";

import { create } from "zustand";

interface UiState {
  configOpen: boolean;
  setConfigOpen: (v: boolean) => void;
  toggleConfig: () => void;
}

export const useUiStore = create<UiState>((set) => ({
  configOpen: false,
  setConfigOpen: (v) => set({ configOpen: v }),
  toggleConfig: () => set((s) => ({ configOpen: !s.configOpen })),
}));
