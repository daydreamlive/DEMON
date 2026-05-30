"use client";

import { useEffect } from "react";

import { listUserUploads } from "@/engine/audio/loadFixture";
import { LOCAL_MODE } from "@/lib/runtime";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { useSessionStore } from "@/store/useSessionStore";

export function useSeedUserUploads(): void {
  const sessionWsUrl = useSessionStore((s) => s.wsUrl);

  useEffect(() => {
    if (!sessionWsUrl && !LOCAL_MODE) return;
    void listUserUploads()
      .then((names) => {
        const store = useCustomTracksStore.getState();
        for (const name of names) {
          if (!store.has(name)) store.addPersisted(name);
        }
      })
      .catch(() => {});
  }, [sessionWsUrl]);
}
