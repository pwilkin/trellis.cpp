// Persistent gallery of past generations, backed by IndexedDB. GLBs are
// 10–15 MB each — far over the ~5 MB localStorage cap — so blobs live here.

import type { GenRecord } from "./types";

const DB_NAME = "trellis-studio";
const DB_VERSION = 1;
const STORE = "generations";

let dbp: Promise<IDBDatabase> | null = null;

function db(): Promise<IDBDatabase> {
  if (dbp) return dbp;
  dbp = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const d = req.result;
      if (!d.objectStoreNames.contains(STORE)) {
        const os = d.createObjectStore(STORE, { keyPath: "id" });
        os.createIndex("ts", "ts");
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbp;
}

function tx(mode: IDBTransactionMode): Promise<IDBObjectStore> {
  return db().then((d) => d.transaction(STORE, mode).objectStore(STORE));
}

function wrap<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export function newId(): string {
  // Not crypto-sensitive; unique enough for gallery keys.
  return `${Date.now().toString(36)}-${Math.floor(Math.random() * 1e9).toString(36)}`;
}

export async function put(rec: GenRecord): Promise<void> {
  await wrap((await tx("readwrite")).put(rec));
}

export async function all(): Promise<GenRecord[]> {
  const recs = await wrap((await tx("readonly")).getAll() as IDBRequest<GenRecord[]>);
  return recs.sort((a, b) => b.ts - a.ts);
}

export async function get(id: string): Promise<GenRecord | undefined> {
  return wrap((await tx("readonly")).get(id) as IDBRequest<GenRecord | undefined>);
}

export async function del(id: string): Promise<void> {
  await wrap((await tx("readwrite")).delete(id));
}

export async function clear(): Promise<void> {
  await wrap((await tx("readwrite")).clear());
}
