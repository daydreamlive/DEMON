import { podHttp } from "@/engine/podUrl";

/** List videos available in the pod's static/videos/ directory. */
export async function listVideos(): Promise<string[]> {
  const res = await fetch(podHttp("/api/videos"));
  if (!res.ok) throw new Error(`/api/videos failed: ${res.status}`);
  const json = (await res.json()) as string[];
  return json;
}
