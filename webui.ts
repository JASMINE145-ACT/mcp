import React, { useEffect, useMemo, useRef, useState, useCallback } from "react";

/**
 * AI Data Agent – Web UI (single-file React app)
 * -------------------------------------------------
 * Goals
 * - Talk to a backend that runs your LangGraph multi-model agent
 * - Upload CSV, preview schema, ask questions
 * - Show the 4 pipeline stages: Plan → Code → Execute → Validate
 * - Display charts (png) and textual results
 * - Designed for easy future upgrades (new tools/models/panels)
 *
 * Backend contract (aligns with your agent in agent.py):
 *  POST /api/upload           -> { file }                          => { datasetId, profile }
 *  GET  /api/profile?id=...   ->                                   => { profile }
 *  POST /api/analyze          -> { question, datasetId }           => {
 *      question, plan, code, execution_result, validation,
 *      images: string[] (filenames or data URLs),
 *      logs?: string[]
 *  }
 *  GET  /api/image?name=...   -> returns the image (e.g., analysis_chart.png)
 *
 * You can adapt the endpoints to your actual FastAPI/Flask app.
 *
 * Extensibility hooks:
 * - registerTool(), registerModel() patterns (see bottom)
 * - panels[] array makes it trivial to add new steps (e.g., "Safety", "Eval")
 * - event bus (simple pub/sub) so future agents can emit custom events
 */

// ---- Minimal Tailwind reset ----
const Shell: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="min-h-screen bg-gray-50 text-gray-900">
    <div className="max-w-7xl mx-auto p-4 md:p-6">{children}</div>
  </div>
);

// ---- Types ----
interface ProfileSummary {
  columns: Array<{ name: string; dtype: string; example?: string }>
  rows?: number
  [k: string]: any
}

interface RunResult {
  question: string
  plan: string
  code: string
  execution_result: string
  validation: string
  images?: string[]
  logs?: string[]
}

interface UploadProgress {
  loaded: number
  total: number
  percentage: number
}

interface ErrorState {
  message: string
  details?: string
  timestamp: Date
}

// ---- Utilities ----
function classNames(...xs: Array<string | undefined | false>) {
  return xs.filter(Boolean).join(" ");
}

function useEventBus<T = any>() {
  const subs = useRef(new Set<(p: T) => void>());
  return useMemo(
    () => ({
      emit: (payload: T) => subs.current.forEach((fn) => fn(payload)),
      on: (fn: (p: T) => void) => {
        subs.current.add(fn);
        return () => subs.current.delete(fn);
      },
    }),
    []
  );
}

// ---- Components ----
const Section: React.FC<{ title: string; right?: React.ReactNode } & React.HTMLAttributes<HTMLDivElement>> = ({ title, right, className, children }) => (
  <div className={classNames("bg-white rounded-2xl shadow-sm border p-4 md:p-6", className)}>
    <div className="flex items-center justify-between mb-4">
      <h2 className="text-lg font-semibold tracking-tight">{title}</h2>
      {right}
    </div>
    {children}
  </div>
);

const TextArea: React.FC<React.TextareaHTMLAttributes<HTMLTextAreaElement>> = (props) => (
  <textarea
    {...props}
    className={classNames(
      "w-full rounded-xl border border-gray-200 px-4 py-3 outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent bg-white transition-all duration-200 resize-none",
      props.disabled && "bg-gray-50 cursor-not-allowed",
      props.className
    )}
  />
);

const Button: React.FC<React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'secondary' | 'danger'; size?: 'sm' | 'md' | 'lg'; loading?: boolean }> = ({ 
  variant = 'primary', 
  size = 'md', 
  loading = false, 
  children, 
  className, 
  disabled,
  ...props 
}) => {
  const baseClasses = "inline-flex items-center justify-center font-medium rounded-xl transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2";
  
  const variantClasses = {
    primary: "bg-indigo-600 hover:bg-indigo-700 text-white focus:ring-indigo-500 shadow-sm",
    secondary: "bg-gray-100 hover:bg-gray-200 text-gray-900 focus:ring-gray-500",
    danger: "bg-red-600 hover:bg-red-700 text-white focus:ring-red-500 shadow-sm"
  };
  
  const sizeClasses = {
    sm: "px-3 py-1.5 text-sm",
    md: "px-4 py-2 text-sm",
    lg: "px-6 py-3 text-base"
  };
  
  return (
    <button
      {...props}
      disabled={disabled || loading}
      className={classNames(
        baseClasses,
        variantClasses[variant],
        sizeClasses[size],
        (disabled || loading) && "opacity-50 cursor-not-allowed",
        className
      )}
    >
      {loading && (
        <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
      )}
      {children}
    </button>
  );
};

const Alert: React.FC<{ type: 'success' | 'error' | 'warning' | 'info'; title?: string; children: React.ReactNode; onClose?: () => void }> = ({ 
  type, 
  title, 
  children, 
  onClose 
}) => {
  const typeClasses = {
    success: "bg-green-50 border-green-200 text-green-800",
    error: "bg-red-50 border-red-200 text-red-800",
    warning: "bg-yellow-50 border-yellow-200 text-yellow-800",
    info: "bg-blue-50 border-blue-200 text-blue-800"
  };
  
  const iconClasses = {
    success: "text-green-400",
    error: "text-red-400",
    warning: "text-yellow-400",
    info: "text-blue-400"
  };
  
  return (
    <div className={classNames("rounded-xl border p-4", typeClasses[type])}>
      <div className="flex">
        <div className="flex-shrink-0">
          {type === 'success' && (
            <svg className={classNames("h-5 w-5", iconClasses[type])} viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
          )}
          {type === 'error' && (
            <svg className={classNames("h-5 w-5", iconClasses[type])} viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
          )}
          {type === 'warning' && (
            <svg className={classNames("h-5 w-5", iconClasses[type])} viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          )}
          {type === 'info' && (
            <svg className={classNames("h-5 w-5", iconClasses[type])} viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
          )}
        </div>
        <div className="ml-3 flex-1">
          {title && <h3 className="text-sm font-medium">{title}</h3>}
          <div className={classNames("text-sm", title && "mt-1")}>{children}</div>
        </div>
        {onClose && (
          <div className="ml-auto pl-3">
            <button
              onClick={onClose}
              className="inline-flex rounded-md p-1.5 hover:bg-black/5 focus:outline-none focus:ring-2 focus:ring-offset-2"
            >
              <svg className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

const ProgressBar: React.FC<{ progress: number; className?: string }> = ({ progress, className }) => (
  <div className={classNames("w-full bg-gray-200 rounded-full h-2", className)}>
    <div 
      className="bg-indigo-600 h-2 rounded-full transition-all duration-300 ease-out"
      style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
    />
  </div>
);

const CodeBlock: React.FC<{ title?: string; text: string; lang?: string }> = ({ title, text }) => (
  <div>
    {title && <div className="text-sm text-gray-500 mb-2">{title}</div>}
    <pre className="bg-gray-900 text-gray-100 rounded-xl p-4 overflow-auto text-sm">
      <code>{text || "(empty)"}</code>
    </pre>
  </div>
);

const Pill: React.FC<{ children: React.ReactNode; color?: string }> = ({ children, color = "indigo" }) => (
  <span className={`inline-flex items-center rounded-full bg-${color}-50 text-${color}-700 ring-1 ring-${color}-200 px-2.5 py-0.5 text-xs font-medium`}>{children}</span>
);

// ---- Main App ----
const App: React.FC = () => {
  const bus = useEventBus<{ type: string; [k: string]: any }>();

  // Dataset state
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [profile, setProfile] = useState<ProfileSummary | null>(null);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress | null>(null);

  // Chat / query
  const [question, setQuestion] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [runProgress, setRunProgress] = useState(0);

  // Results
  const [result, setResult] = useState<RunResult | null>(null);
  const [activePanel, setActivePanel] = useState("Plan");
  
  // Error handling
  const [error, setError] = useState<ErrorState | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const clearMessages = useCallback(() => {
    setError(null);
    setSuccess(null);
  }, []);

  const showError = useCallback((message: string, details?: string) => {
    setError({ message, details, timestamp: new Date() });
    setSuccess(null);
  }, []);

  const showSuccess = useCallback((message: string) => {
    setSuccess(message);
    setError(null);
  }, []);

  async function uploadFile(file: File) {
    if (!file) return;
    
    // Validate file type
    if (!file.name.toLowerCase().endsWith('.csv')) {
      showError('Invalid file type', 'Please upload a CSV file.');
      return;
    }
    
    // Validate file size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      showError('File too large', 'Please upload a file smaller than 10MB.');
      return;
    }
    
    clearMessages();
    setUploadProgress({ loaded: 0, total: file.size, percentage: 0 });
    
    try {
      const fd = new FormData();
      fd.append("file", file);
      
      const xhr = new XMLHttpRequest();
      
      // Track upload progress
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          const percentage = Math.round((e.loaded / e.total) * 100);
          setUploadProgress({ loaded: e.loaded, total: e.total, percentage });
        }
      });
      
      const response = await new Promise<Response>((resolve, reject) => {
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(new Response(xhr.responseText, { status: xhr.status }));
          } else {
            reject(new Error(`Upload failed with status ${xhr.status}`));
          }
        };
        xhr.onerror = () => reject(new Error('Network error during upload'));
        xhr.open('POST', '/api/upload');
        xhr.send(fd);
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Upload failed');
      }
      
      setDatasetId(data.datasetId);
      setProfile(data.profile);
      setUploadProgress(null);
      showSuccess(`Successfully uploaded ${file.name}`);
      bus.emit({ type: "dataset:loaded", datasetId: data.datasetId });
      
    } catch (error: any) {
      setUploadProgress(null);
      showError('Upload failed', error.message || 'An unexpected error occurred');
    }
  }

  async function runAnalysis() {
    if (!datasetId) {
      showError('No dataset', 'Please upload a dataset first.');
      return;
    }
    if (!question.trim()) {
      showError('No question', 'Please enter a question.');
      return;
    }
    
    clearMessages();
    setIsRunning(true);
    setResult(null);
    setRunProgress(0);
    
    // Simulate progress updates
    const progressInterval = setInterval(() => {
      setRunProgress(prev => Math.min(prev + Math.random() * 15, 90));
    }, 500);
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minute timeout
      
      const r = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim(), datasetId }),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!r.ok) {
        const errorData = await r.json().catch(() => ({}));
        throw new Error(errorData.error || `Analysis failed with status ${r.status}`);
      }
      
      const data: RunResult = await r.json();
      
      if (!data || typeof data !== 'object') {
        throw new Error('Invalid response format');
      }
      
      setResult(data);
      setActivePanel("Plan");
      setRunProgress(100);
      showSuccess('Analysis completed successfully!');
      bus.emit({ type: "run:done", data });
      
    } catch (error: any) {
      if (error.name === 'AbortError') {
        showError('Analysis timeout', 'The analysis took too long to complete. Please try with a simpler question.');
      } else {
        showError('Analysis failed', error.message || 'An unexpected error occurred during analysis.');
      }
    } finally {
      clearInterval(progressInterval);
      setIsRunning(false);
      setTimeout(() => setRunProgress(0), 2000);
    }
  }

  return (
    <Shell>
      {/* Global Messages */}
      <div className="mb-6 space-y-3">
        {error && (
          <Alert type="error" title={error.message} onClose={clearMessages}>
            {error.details && <div className="mt-1 text-sm">{error.details}</div>}
          </Alert>
        )}
        {success && (
          <Alert type="success" onClose={clearMessages}>
            {success}
          </Alert>
        )}
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* LEFT: Dataset & Settings */}
        <div className="space-y-6 lg:col-span-1">
          <Section title="Dataset Upload">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Upload CSV File
                </label>
                <div className="relative">
                  <input
                    type="file"
                    accept=".csv,text/csv"
                    onChange={(e) => e.target.files && uploadFile(e.target.files[0])}
                    disabled={uploadProgress !== null}
                    className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100 disabled:opacity-50 disabled:cursor-not-allowed"
                  />
                </div>
                {uploadProgress && (
                  <div className="mt-3 space-y-2">
                    <div className="flex justify-between text-sm text-gray-600">
                      <span>Uploading...</span>
                      <span>{uploadProgress.percentage}%</span>
                    </div>
                    <ProgressBar progress={uploadProgress.percentage} />
                  </div>
                )}
              </div>
              
              {profile && (
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <Pill>Columns: {profile.columns?.length ?? "-"}</Pill>
                    {profile.rows != null && <Pill color="emerald">Rows: {profile.rows}</Pill>}
                  </div>
                  <div className="max-h-64 overflow-auto border border-gray-200 rounded-xl">
                    <table className="w-full text-sm">
                      <thead className="sticky top-0">
                        <tr className="bg-gray-50 border-b border-gray-200">
                          <th className="text-left px-3 py-2 font-medium text-gray-900">Name</th>
                          <th className="text-left px-3 py-2 font-medium text-gray-900">Type</th>
                          <th className="text-left px-3 py-2 font-medium text-gray-900">Example</th>
                        </tr>
                      </thead>
                      <tbody>
                        {profile.columns?.map((c, i) => (
                          <tr key={i} className="border-t border-gray-100 hover:bg-gray-50">
                            <td className="px-3 py-2 font-medium text-gray-900">{c.name}</td>
                            <td className="px-3 py-2 text-gray-600">
                              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800">
                                {c.dtype}
                              </span>
                            </td>
                            <td className="px-3 py-2 text-gray-600 truncate max-w-32" title={c.example}>
                              {c.example ?? "—"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </Section>

          <Section title="Analysis Query">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  What would you like to analyze?
                </label>
                <TextArea
                  rows={4}
                  placeholder="e.g., Compare average rating by category and plot a bar chart"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  disabled={isRunning || !datasetId}
                />
                <div className="mt-2 text-xs text-gray-500">
                  Try asking about trends, comparisons, or visualizations
                </div>
              </div>
              
              {isRunning && runProgress > 0 && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm text-gray-600">
                    <span>Analyzing...</span>
                    <span>{Math.round(runProgress)}%</span>
                  </div>
                  <ProgressBar progress={runProgress} />
                </div>
              )}
              
              <Button
                onClick={runAnalysis}
                disabled={isRunning || !datasetId || !question.trim()}
                loading={isRunning}
                size="lg"
                className="w-full"
              >
                {isRunning ? "Analyzing..." : "Run Analysis"}
              </Button>
            </div>
          </Section>

          <Section title="System Logs">
            <div className="bg-gray-900 rounded-xl p-4 min-h-[8rem] max-h-48 overflow-auto">
              <div className="text-xs font-mono text-gray-300 whitespace-pre-wrap">
                {result?.logs?.length ? (
                  result.logs.map((log, i) => (
                    <div key={i} className="mb-1">
                      <span className="text-gray-500">[{new Date().toLocaleTimeString()}]</span> {log}
                    </div>
                  ))
                ) : (
                  <div className="text-gray-500 italic">No logs available</div>
                )}
              </div>
            </div>
          </Section>
        </div>

        {/* RIGHT: Results */}
        <div className="space-y-6 lg:col-span-3">
          <Section
            title="Analysis Pipeline"
            right={
              result && (
                <div className="flex gap-1 bg-gray-100 rounded-lg p-1">
                  {(["Plan", "Code", "Execute", "Validate"] as const).map((p) => {
                    const isActive = activePanel === p;
                    const hasContent = result && (
                      (p === "Plan" && result.plan) ||
                      (p === "Code" && result.code) ||
                      (p === "Execute" && result.execution_result) ||
                      (p === "Validate" && result.validation)
                    );
                    
                    return (
                      <button
                        key={p}
                        onClick={() => setActivePanel(p)}
                        disabled={!hasContent}
                        className={classNames(
                          "px-3 py-1.5 rounded-md text-sm font-medium transition-all duration-200",
                          isActive 
                            ? "bg-white text-indigo-600 shadow-sm" 
                            : hasContent 
                              ? "text-gray-700 hover:text-gray-900 hover:bg-gray-50" 
                              : "text-gray-400 cursor-not-allowed",
                          hasContent && "relative"
                        )}
                      >
                        {p}
                        {hasContent && (
                          <span className={classNames(
                            "absolute -top-1 -right-1 h-2 w-2 rounded-full",
                            isActive ? "bg-indigo-600" : "bg-green-400"
                          )} />
                        )}
                      </button>
                    );
                  })}
                </div>
              )
            }
          >
            {!result ? (
              <div className="text-center py-12">
                <div className="mx-auto h-12 w-12 text-gray-400 mb-4">
                  <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Analysis Yet</h3>
                <p className="text-gray-500 mb-4">Upload a dataset and ask a question to see the analysis pipeline in action.</p>
                <div className="flex justify-center space-x-8 text-sm text-gray-400">
                  <div className="flex items-center space-x-2">
                    <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center text-xs font-medium">1</div>
                    <span>Plan</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center text-xs font-medium">2</div>
                    <span>Code</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center text-xs font-medium">3</div>
                    <span>Execute</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center text-xs font-medium">4</div>
                    <span>Validate</span>
                  </div>
                </div>
              </div>
            ) : (
              <>
                {/* Panels */}
                {activePanel === "Plan" && (
                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <CodeBlock title="Analysis Plan" text={result?.plan || "No plan available"} />
                    </div>
                    <div className="bg-blue-50 rounded-xl p-4">
                      <div className="flex items-start space-x-3">
                        <div className="flex-shrink-0">
                          <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                            <svg className="w-4 h-4 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                            </svg>
                          </div>
                        </div>
                        <div>
                          <h4 className="text-sm font-medium text-blue-900 mb-2">Planning Stage</h4>
                          <div className="text-sm text-blue-800 space-y-2">
                            <p>• GPT-5 nano analyzes your question and creates a structured plan</p>
                            <p>• Identifies required data columns and analysis methods</p>
                            <p>• Determines the best approach for visualization or calculation</p>
                            <p>• Sets expectations for the final output format</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {activePanel === "Code" && (
                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <CodeBlock title="Generated Code" text={result?.code || "No code available"} lang="python" />
                    </div>
                    <div className="bg-purple-50 rounded-xl p-4">
                      <div className="flex items-start space-x-3">
                        <div className="flex-shrink-0">
                          <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center">
                            <svg className="w-4 h-4 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                            </svg>
                          </div>
                        </div>
                        <div>
                          <h4 className="text-sm font-medium text-purple-900 mb-2">Code Generation</h4>
                          <div className="text-sm text-purple-800 space-y-2">
                            <p>• Claude Sonnet generates Python code based on the plan</p>
                            <p>• Uses pandas for data manipulation and matplotlib for visualization</p>
                            <p>• Follows secure coding practices with sandboxed execution</p>
                            <p>• Supports both analysis and plotting operations</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {activePanel === "Execute" && (
                  <div className="space-y-6">
                    <div className="grid md:grid-cols-2 gap-6">
                      <div>
                        <CodeBlock title="Execution Output" text={result?.execution_result || "No execution result available"} />
                      </div>
                      <div className="bg-green-50 rounded-xl p-4">
                        <div className="flex items-start space-x-3">
                          <div className="flex-shrink-0">
                            <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                              <svg className="w-4 h-4 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                              </svg>
                            </div>
                          </div>
                          <div>
                            <h4 className="text-sm font-medium text-green-900 mb-2">Code Execution</h4>
                            <div className="text-sm text-green-800 space-y-2">
                              <p>• Code runs in a secure Python environment</p>
                              <p>• Results are captured and formatted for display</p>
                              <p>• Charts are automatically saved and displayed</p>
                              <p>• Errors are caught and reported safely</p>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    {result?.images && result.images.length > 0 && (
                      <div>
                        <h4 className="text-lg font-medium text-gray-900 mb-4">Generated Visualizations</h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                          {result.images.map((img, i) => (
                            <div key={i} className="bg-white rounded-xl border border-gray-200 overflow-hidden shadow-sm hover:shadow-md transition-shadow">
                              <img 
                                src={img.startsWith("data:") ? img : `/api/image?name=${encodeURIComponent(img)}`}
                                alt={`Analysis chart ${i + 1}`} 
                                className="w-full h-48 object-contain bg-gray-50"
                                onError={(e) => {
                                  const target = e.target as HTMLImageElement;
                                  target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjNmNGY2Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzZiNzI4MCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkltYWdlIG5vdCBmb3VuZDwvdGV4dD48L3N2Zz4=';
                                }}
                              />
                              <div className="p-3">
                                <p className="text-sm text-gray-600">Chart {i + 1}</p>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {activePanel === "Validate" && (
                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <CodeBlock title="Validation & Insights" text={result?.validation || "No validation available"} />
                    </div>
                    <div className="bg-amber-50 rounded-xl p-4">
                      <div className="flex items-start space-x-3">
                        <div className="flex-shrink-0">
                          <div className="w-8 h-8 bg-amber-100 rounded-full flex items-center justify-center">
                            <svg className="w-4 h-4 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                          </div>
                        </div>
                        <div>
                          <h4 className="text-sm font-medium text-amber-900 mb-2">Validation & Review</h4>
                          <div className="text-sm text-amber-800 space-y-2">
                            <p>• GPT-4o mini reviews the analysis for accuracy</p>
                            <p>• Provides plain-language explanations of results</p>
                            <p>• Suggests improvements and next steps</p>
                            <p>• Validates the approach and identifies potential issues</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
          </Section>

          {result && (
            <Section title="Export & Actions">
              <div className="flex flex-wrap gap-3">
                <Button 
                  variant="secondary" 
                  onClick={() => {
                    const data = {
                      question: result.question,
                      timestamp: new Date().toISOString(),
                      results: result
                    };
                    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `analysis-${Date.now()}.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                >
                  <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Export Results
                </Button>
                
                <Button 
                  variant="secondary"
                  onClick={() => {
                    setResult(null);
                    setQuestion("");
                    setActivePanel("Plan");
                  }}
                >
                  <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  New Analysis
                </Button>
                
                {result.images && result.images.length > 0 && (
                  <Button 
                    variant="secondary"
                    onClick={async () => {
                      try {
                        const zip = await import('jszip');
                        const JSZip = zip.default;
                        const zipFile = new JSZip();
                        
                        for (let i = 0; i < result.images!.length; i++) {
                          const img = result.images![i];
                          if (img.startsWith('data:')) {
                            const base64Data = img.split(',')[1];
                            zipFile.file(`chart-${i + 1}.png`, base64Data, { base64: true });
                          }
                        }
                        
                        const content = await zipFile.generateAsync({ type: 'blob' });
                        const url = URL.createObjectURL(content);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `charts-${Date.now()}.zip`;
                        a.click();
                        URL.revokeObjectURL(url);
                      } catch (error) {
                        showError('Export failed', 'Could not export charts. Please try downloading them individually.');
                      }
                    }}
                  >
                    <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    Download Charts
                  </Button>
                )}
              </div>
            </Section>
          )}
          
          <Section title="About This Tool" className="mb-12">
            <div className="space-y-4 text-sm text-gray-600">
              <div>
                <h4 className="font-medium text-gray-900 mb-2">AI-Powered Data Analysis</h4>
                <p>This tool uses a multi-model AI pipeline to analyze your data:</p>
                <div className="mt-2 space-y-1 text-xs">
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                    <span><strong>GPT-5 Nano:</strong> Creates analysis plans</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                    <span><strong>Claude Sonnet:</strong> Generates Python code</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                    <span><strong>Secure Runtime:</strong> Executes code safely</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-amber-400 rounded-full"></div>
                    <span><strong>GPT-4o Mini:</strong> Validates and explains results</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h4 className="font-medium text-gray-900 mb-2">Supported Features</h4>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>• Statistical analysis</div>
                  <div>• Data visualization</div>
                  <div>• Trend analysis</div>
                  <div>• Comparative studies</div>
                  <div>• Chart generation</div>
                  <div>• Export capabilities</div>
                </div>
              </div>
            </div>
          </Section>
        </div>
      </div>
    </Shell>
  );
};

export default App;

// Auto-clear messages after 5 seconds
setTimeout(() => {
  if (typeof window !== 'undefined') {
    const style = document.createElement('style');
    style.textContent = `
      .auto-dismiss {
        animation: fadeOut 0.5s ease-in-out 4.5s forwards;
      }
      
      @keyframes fadeOut {
        from { opacity: 1; transform: translateY(0); }
        to { opacity: 0; transform: translateY(-10px); }
      }
    `;
    document.head.appendChild(style);
  }
}, 100);

// ----------------- Extension APIs -----------------
export type Tool = { 
  name: string; 
  desc: string; 
  run: (args: any) => Promise<any>;
  category?: 'analysis' | 'visualization' | 'export' | 'other';
};

export const toolRegistry: Record<string, Tool> = {};

export function registerTool(tool: Tool) { 
  toolRegistry[tool.name] = tool; 
}

export type Model = { 
  id: string; 
  provider: "openai" | "anthropic" | "other"; 
  role: "planner" | "executor" | "validator";
  description?: string;
};

export const modelRegistry: Model[] = [
  { 
    id: "gpt-5-nano", 
    provider: "openai", 
    role: "planner",
    description: "Creates structured analysis plans"
  },
  { 
    id: "claude-sonnet-4-5", 
    provider: "anthropic", 
    role: "executor",
    description: "Generates Python code for data analysis"
  },
  { 
    id: "gpt-4o-mini", 
    provider: "openai", 
    role: "validator",
    description: "Validates results and provides insights"
  },
];

// Event system for extensibility
export interface AnalysisEvent {
  type: 'dataset:loaded' | 'analysis:started' | 'analysis:completed' | 'error:occurred';
  payload?: any;
  timestamp: Date;
}

export function createAnalysisEvent(type: AnalysisEvent['type'], payload?: any): AnalysisEvent {
  return {
    type,
    payload,
    timestamp: new Date()
  };
}
