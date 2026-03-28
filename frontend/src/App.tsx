// Purpose: ICU dashboard UI with realtime camera monitoring and fall alert sidebar.
import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import { analyzeFrame } from "./api";
import type { AnalyzeRequest, DetectionResult, Point } from "./types";

const CAMERA_CAPTURE_INTERVAL_MS = 2000;

/** COCO-17 limb pairs for YOLO pose overlay. */
const POSE_EDGES: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
  [5, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
] as const;
const CAMERA_CONSTRAINTS: MediaStreamConstraints[] = [
  { video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
  { video: { facingMode: { ideal: "user" }, width: { ideal: 960 }, height: { ideal: 540 } }, audio: false },
  { video: true, audio: false },
];

function App() {
  const [patientId, setPatientId] = useState<string>("icu-patient-001");
  const [areaName, setAreaName] = useState<string>("icu_bed_1");
  const [status, setStatus] = useState<string>("Waiting for frame.");
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isCameraOn, setIsCameraOn] = useState<boolean>(false);
  const [isRealtimeOn, setIsRealtimeOn] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string>("");
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const realtimeTimerRef = useRef<number | null>(null);
  const isRealtimeBusyRef = useRef<boolean>(false);

  const requestPayload = useMemo<AnalyzeRequest>(
    () => ({
      patient_id: patientId ?? "unknown",
      area_name: areaName ?? "icu_bed_1",
      image_base64: "",
    }),
    [areaName, patientId],
  );

  const handleAnalyzeClick = async () => {
    const frame = captureCameraFrame();
    if (!frame) return setStatus("No camera frame available yet.");
    await runAnalyzeRequest(frame);
  };

  const startCamera = async () => {
    if (isCameraOn) return;
    setCameraError("");
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError("Camera API not available in this browser.");
      setStatus("Camera not supported.");
      return;
    }
    if (!window.isSecureContext) {
      setCameraError("Camera needs secure context. Use localhost or HTTPS.");
      setStatus("Camera blocked by browser security.");
      return;
    }
    try {
      const stream = await getCameraStream();
      if (!stream) {
        setCameraError("Camera stream unavailable.");
        setStatus("Camera stream unavailable.");
        return;
      }
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setIsCameraOn(true);
      setStatus("Camera connected.");
    } catch (error: unknown) {
      const message = getCameraErrorMessage(error);
      setCameraError(message);
      setStatus("Camera connection failed.");
      setIsCameraOn(false);
    }
  };

  const stopCamera = () => {
    stopRealtimeMonitoring();
    streamRef.current?.getTracks()?.forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsCameraOn(false);
    setStatus("Camera stopped.");
  };

  const captureCameraFrame = (): string => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.videoWidth < 1 || video.videoHeight < 1) return "";
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) return "";
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.8) ?? "";
  };

  const handleCaptureFromCamera = () => {
    const captured = captureCameraFrame();
    if (!captured) {
      setStatus("No camera frame available yet.");
      return;
    }
    setStatus("Captured frame from live camera. Running analysis...");
    void runAnalyzeRequest(captured);
  };

  const startRealtimeMonitoring = () => {
    if (!isCameraOn) {
      setStatus("Start camera before realtime monitoring.");
      return;
    }
    if (isRealtimeOn) return;
    setIsRealtimeOn(true);
    setStatus("Realtime monitoring started.");
    realtimeTimerRef.current = window.setInterval(async () => {
      if (isRealtimeBusyRef.current) return;
      isRealtimeBusyRef.current = true;
      const currentFrame = captureCameraFrame();
      if (currentFrame) await runAnalyzeRequest(currentFrame);
      isRealtimeBusyRef.current = false;
    }, CAMERA_CAPTURE_INTERVAL_MS);
  };

  const stopRealtimeMonitoring = () => {
    if (realtimeTimerRef.current !== null) {
      window.clearInterval(realtimeTimerRef.current);
      realtimeTimerRef.current = null;
    }
    setIsRealtimeOn(false);
  };

  const runAnalyzeRequest = async (frameBase64: string) => {
    setIsLoading(true);
    setStatus("Analyzing frame...");
    const response = await analyzeFrame({
      ...requestPayload,
      image_base64: frameBase64 ?? "",
    });
    setIsLoading(false);
    if (response.error) {
      setResult(null);
      setStatus(response.error);
      return;
    }
    setResult(response.data);
    setStatus(response.data?.message ?? "Analysis completed.");
  };

  useEffect(() => {
    return () => {
      stopRealtimeMonitoring();
      streamRef.current?.getTracks()?.forEach((track) => track.stop());
    };
  }, []);

  return (
    <main className="container">
      <h1>ICU Fall Detection</h1>
      <p className="subtitle">Realtime Camera + Human Fall Alerts</p>
      <div className="dashboard">
        <section className="monitorCard">
          <div className="videoWrap">
            <video ref={videoRef} className="cameraPreview" autoPlay muted playsInline />
            <svg
              className="overlay"
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
            >
              {(result?.people ?? []).map((person) => {
                const kp = person.keypoints ?? [];
                const groupClass =
                  person.status === "FALL_DETECTED" ? "poseGroup poseGroupFall" : "poseGroup poseGroupOk";
                const labelAnchor = kp[0] && isKeypointVisible(kp[0]) ? kp[0] : kp[5] ?? kp[6];
                return (
                  <g key={`person-${person.person_id}`} className={groupClass}>
                    {POSE_EDGES.map(([a, b], edgeIndex) => {
                      const pa = kp[a];
                      const pb = kp[b];
                      if (!isKeypointVisible(pa) || !isKeypointVisible(pb)) return null;
                      return (
                        <line
                          key={`sk-${person.person_id}-${edgeIndex}`}
                          x1={pa.x * 100}
                          y1={pa.y * 100}
                          x2={pb.x * 100}
                          y2={pb.y * 100}
                          className="skeletonLine"
                        />
                      );
                    })}
                    {kp.map((joint, jointIndex) =>
                      isKeypointVisible(joint) ? (
                        <circle
                          key={`jt-${person.person_id}-${jointIndex}`}
                          cx={joint.x * 100}
                          cy={joint.y * 100}
                          r={0.85}
                          className="skeletonJoint"
                        />
                      ) : null,
                    )}
                    {labelAnchor && isKeypointVisible(labelAnchor) ? (
                      <text
                        x={labelAnchor.x * 100}
                        y={labelAnchor.y * 100 - 2}
                        className="poseLabel"
                      >
                        P{person.person_id} {person.status}
                      </text>
                    ) : null}
                  </g>
                );
              })}
            </svg>
          </div>
          <canvas ref={canvasRef} className="hiddenCanvas" />
          <div className="cameraActions">
            <button onClick={startCamera} disabled={isCameraOn}>Start Camera</button>
            <button onClick={stopCamera} disabled={!isCameraOn}>Stop Camera</button>
            <button onClick={handleCaptureFromCamera} disabled={!isCameraOn || isLoading}>Analyze Now</button>
            <button onClick={startRealtimeMonitoring} disabled={!isCameraOn || isRealtimeOn}>Start Realtime</button>
            <button onClick={stopRealtimeMonitoring} disabled={!isRealtimeOn}>Stop Realtime</button>
            <button onClick={handleAnalyzeClick} disabled={!isCameraOn || isLoading}>Analyze Current Frame</button>
          </div>
        </section>

        <aside className="sidebar">
          <section className={`card alertCard ${result?.status ?? "NO_PERSON"}`}>
            <h2>Live Alert</h2>
            <p className="alertMessage">{result?.message ?? status}</p>
            <ul>
              <li>Status: {result?.status ?? "NO_PERSON"}</li>
              <li>Confidence: {(result?.confidence ?? 0).toFixed(2)}</li>
              <li>People: {result?.person_count ?? 0}</li>
              <li>Device: {result?.used_cuda ? `CUDA (${result?.device ?? "cuda:0"})` : "CPU"}</li>
            </ul>
            {(result?.people?.length ?? 0) > 0 ? (
              <ul className="peopleList">
                {(result?.people ?? []).slice(0, 5).map((person) => (
                  <li key={`person-row-${person.person_id}`}>
                    P{person.person_id}: {person.status} ({person.confidence.toFixed(2)})
                  </li>
                ))}
              </ul>
            ) : null}
          </section>
          <section className="card">
            <h2>Patient Setup</h2>
            <label>
              Patient ID
              <input value={patientId} onChange={(event) => setPatientId(event.target.value ?? "")} />
            </label>
            <label>
              Area Name
              <input value={areaName} onChange={(event) => setAreaName(event.target.value ?? "")} />
            </label>
            {cameraError ? <p className="error">{cameraError}</p> : null}
            <p className="note">{status}</p>
          </section>
        </aside>
      </div>
    </main>
  );
}

export default App;

function isKeypointVisible(p: Point | undefined): boolean {
  if (!p) return false;
  return p.x > 0.005 || p.y > 0.005;
}

async function getCameraStream(): Promise<MediaStream | null> {
  for (const constraints of CAMERA_CONSTRAINTS) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (stream) return stream;
    } catch {
      continue;
    }
  }
  return null;
}

function getCameraErrorMessage(error: unknown): string {
  if (!(error instanceof DOMException)) {
    return "Unable to access camera. Please check permissions.";
  }
  if (error.name === "NotAllowedError" || error.name === "SecurityError") {
    return "Camera permission denied. Allow camera access and reload.";
  }
  if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
    return "No camera detected on this device.";
  }
  if (error.name === "NotReadableError" || error.name === "TrackStartError") {
    return "Camera is busy in another app. Close it and retry.";
  }
  if (error.name === "OverconstrainedError" || error.name === "ConstraintNotSatisfiedError") {
    return "Camera mode not supported. Retrying with basic mode failed.";
  }
  return "Unable to access camera. Please check permissions.";
}
