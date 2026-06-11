import { create } from "zustand";

import type { WireContract } from "@demon/client";

// Holds the backend WebSocket wire contract fetched at boot
// (GET /api/protocol). Seeded empty; RTMGBoot fills it. Any surface that
// builds control messages or decodes server events against the backend-owned
// vocabulary reads from here. Kept separate from useKnobManifestStore: that
// store is the `params` payload schema, this one is the command/event
// envelope vocabulary.
interface WireContractState {
  contract: WireContract | null;
  loaded: boolean;
  setContract: (contract: WireContract) => void;
}

export const useWireContractStore = create<WireContractState>((set) => ({
  contract: null,
  loaded: false,
  setContract: (contract) => set({ contract, loaded: true }),
}));
