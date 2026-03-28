// Purpose: API helpers with timeout and graceful fallbacks for dashboard requests.
import type { AnalyzeRequest, ApiResult, DetectionResult } from "./types";

const API_URL = "/api/v1/analyze";
const REQUEST_TIMEOUT_MS = 12000;

function createTimeoutSignal(ms: number = REQUEST_TIMEOUT_MS): AbortSignal {
  const controller = new AbortController();
  setTimeout(() => controller.abort(), ms);
  return controller.signal;
}

export async function analyzeFrame(payload: AnalyzeRequest): Promise<ApiResult<DetectionResult>> {
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: createTimeoutSignal(),
    });
    if (!response.ok) {
      return { data: null, error: "Detection service is temporarily unavailable." };
    }
    const data = (await response.json()) as DetectionResult;
    return { data, error: null };
  } catch (error) {
    console.error("analyzeFrame failed", error);
    return { data: null, error: "Network issue. Please retry in a moment." };
  }
}
