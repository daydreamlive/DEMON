"use client";

import type { DecodedFixture } from "@/engine/audio/loadFixture";

const DB_NAME = "demon-local-session-assets";
const DB_VERSION = 3;
const AUDIO_STORE = "audio";
const FILE_STORE = "files";
const SESSION_STORE = "sessions";

interface AudioAssetRecord extends DecodedFixture {
  id: string;
  updatedAt: number;
}

interface UploadFileRecord {
  id: string;
  fileName: string;
  type: string;
  lastModified: number;
  blob: Blob;
  updatedAt: number;
}

function unavailable(): Error {
  return new Error("Local audio storage is not available in this browser.");
}

function openDb(): Promise<IDBDatabase> {
  if (typeof indexedDB === "undefined") return Promise.reject(unavailable());
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(AUDIO_STORE)) {
        db.createObjectStore(AUDIO_STORE, { keyPath: "id" });
      }
      if (!db.objectStoreNames.contains(FILE_STORE)) {
        db.createObjectStore(FILE_STORE, { keyPath: "id" });
      }
      if (!db.objectStoreNames.contains(SESSION_STORE)) {
        db.createObjectStore(SESSION_STORE, { keyPath: "id" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? unavailable());
  });
}

async function withStore<T>(
  storeName: string,
  mode: IDBTransactionMode,
  run: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, mode);
    const req = run(tx.objectStore(storeName));
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error("Audio asset operation failed."));
    tx.oncomplete = () => db.close();
    tx.onabort = () => {
      db.close();
      reject(tx.error ?? new Error("Audio asset transaction aborted."));
    };
  });
}

export async function putSessionAudioAsset(
  id: string,
  decoded: DecodedFixture,
): Promise<void> {
  const record: AudioAssetRecord = {
    id,
    interleaved: decoded.interleaved,
    channels: decoded.channels,
    frames: decoded.frames,
    sampleRate: decoded.sampleRate,
    updatedAt: Date.now(),
  };
  await withStore(AUDIO_STORE, "readwrite", (store) => store.put(record));
}

export async function getSessionAudioAsset(
  id: string,
): Promise<DecodedFixture | null> {
  const record = await withStore<AudioAssetRecord | undefined>(
    AUDIO_STORE,
    "readonly",
    (store) => store.get(id),
  );
  if (!record) return null;
  return {
    interleaved: record.interleaved,
    channels: record.channels,
    frames: record.frames,
    sampleRate: record.sampleRate,
  };
}

export async function hasSessionAudioAsset(id: string): Promise<boolean> {
  const count = await withStore<number>(AUDIO_STORE, "readonly", (store) =>
    store.count(id),
  );
  return count > 0;
}

export async function deleteSessionAudioAsset(id: string): Promise<void> {
  await withStore(AUDIO_STORE, "readwrite", (store) => store.delete(id));
}

export async function putSessionUploadFile(id: string, file: File): Promise<void> {
  const record: UploadFileRecord = {
    id,
    fileName: file.name,
    type: file.type,
    lastModified: file.lastModified,
    blob: file,
    updatedAt: Date.now(),
  };
  await withStore(FILE_STORE, "readwrite", (store) => store.put(record));
}

export async function getSessionUploadFile(id: string): Promise<File | null> {
  const record = await withStore<UploadFileRecord | undefined>(
    FILE_STORE,
    "readonly",
    (store) => store.get(id),
  );
  if (!record) return null;
  return new File([record.blob], record.fileName, {
    type: record.type,
    lastModified: record.lastModified,
  });
}

export async function hasSessionUploadFile(id: string): Promise<boolean> {
  const count = await withStore<number>(FILE_STORE, "readonly", (store) =>
    store.count(id),
  );
  return count > 0;
}

export async function deleteSessionUploadFile(id: string): Promise<void> {
  await withStore(FILE_STORE, "readwrite", (store) => store.delete(id));
}

export async function putLocalSavedSessionRecord<T extends { id: string }>(
  record: T,
): Promise<void> {
  await withStore(SESSION_STORE, "readwrite", (store) => store.put(record));
}

export async function getLocalSavedSessionRecord<T>(
  id: string,
): Promise<T | null> {
  const record = await withStore<T | undefined>(SESSION_STORE, "readonly", (store) =>
    store.get(id),
  );
  return record ?? null;
}

export async function listLocalSavedSessionRecords<T>(): Promise<T[]> {
  return withStore<T[]>(SESSION_STORE, "readonly", (store) => store.getAll());
}

export async function deleteLocalSavedSessionRecord(id: string): Promise<void> {
  await withStore(SESSION_STORE, "readwrite", (store) => store.delete(id));
}
