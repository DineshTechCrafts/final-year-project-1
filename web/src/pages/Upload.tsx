import { useState, useRef } from 'react';
import { uploadImage, getOverlayUrl, synthesizeResults } from '../api/client';
import { UploadCloud, FileImage, Loader2, BrainCircuit, Activity, MessageSquare, AlertTriangle, Clock } from 'lucide-react';

export default function Upload() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [llmLoading, setLlmLoading] = useState(false);
  const [llmSynthesis, setLlmSynthesis] = useState<any>(null);
  const [llmLatency, setLlmLatency] = useState<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResults(null);
      setError(null);
      setLlmSynthesis(null);
      setLlmLatency(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setLlmSynthesis(null);
    setLlmLatency(null);
    
    let pipelineData = null;
    try {
      pipelineData = await uploadImage(file);
      setResults(pipelineData);
    } catch (err: any) {
      setError(err.message);
      setLoading(false);
      return;
    } 
    setLoading(false);

    // KICK OFF LLM
    setLlmLoading(true);
    try {
      const start = performance.now();
      const llmData = await synthesizeResults(pipelineData);
      const end = performance.now();
      setLlmLatency(end - start);
      setLlmSynthesis(llmData.synthesis);
    } catch (err: any) {
      console.error("LLM failed", err);
    } finally {
      setLlmLoading(false);
    }
  };

  return (
    <div className="p-8 max-w-7xl mx-auto flex flex-col h-full">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold text-white flex items-center gap-3">
          <UploadCloud className="w-8 h-8 text-cyan-400" />
          Upload & Inference
        </h1>
      </div>

      <div className="flex flex-col md:flex-row gap-8">
        {/* Left Side: Upload Box */}
        <div className="w-full md:w-1/3 flex flex-col gap-6">
          <div 
            className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center transition-colors cursor-pointer min-h-[300px]
              ${preview ? 'border-cyan-500/50 bg-slate-900' : 'border-slate-700 bg-slate-900/50 hover:bg-slate-900 hover:border-slate-600'}`}
            onClick={() => fileInputRef.current?.click()}
          >
            <input 
              type="file" 
              className="hidden" 
              ref={fileInputRef} 
              accept="image/png, image/jpeg" 
              onChange={handleFileChange}
            />
            
            {preview ? (
              <div className="relative w-full h-full flex flex-col items-center justify-center">
                <img src={preview} alt="Upload preview" className="max-h-48 object-contain mb-4 rounded" />
                <span className="text-sm font-medium text-slate-300 bg-slate-950 px-3 py-1 rounded-full border border-slate-800">
                  {file?.name}
                </span>
                <p className="text-xs text-slate-500 mt-2">Click to select another file</p>
              </div>
            ) : (
              <>
                <FileImage className="w-12 h-12 text-slate-500 mb-4" />
                <p className="text-lg font-bold text-slate-300">Drag & Drop CT Scan</p>
                <p className="text-sm text-slate-500 mt-2 mb-6">Supports PNG or JPG (2D Slices)</p>
                <div className="bg-slate-800 text-white px-4 py-2 rounded-lg font-medium text-sm">
                  Browse Files
                </div>
              </>
            )}
          </div>

          <button
            onClick={handleUpload}
            disabled={!file || loading}
            className="w-full flex justify-center items-center gap-2 bg-cyan-600 hover:bg-cyan-500 text-white font-bold py-3 px-4 rounded-xl shadow-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <BrainCircuit className="w-5 h-5" />}
            {loading ? 'Processing Pipeline...' : 'Run Analysis Pipeline'}
          </button>
          
          {error && (
            <div className="p-4 bg-red-950/50 border border-red-900 text-red-400 rounded-lg text-sm">
              {error}
            </div>
          )}
          
          {results && (
            <div className="bg-slate-900 p-4 rounded-xl border border-slate-800 shadow-lg">
              <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-3 flex items-center gap-2">
                <Activity className="w-4 h-4 text-green-400" /> Extracted Features
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-400">Mean Intensity</span>
                  <span className="font-mono text-cyan-400">{results.simulated_features.mean_intensity}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Liver Ratio</span>
                  <span className="font-mono text-red-400">{(results.simulated_features.liver_ratio * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Mass Ratio</span>
                  <span className="font-mono text-yellow-400">{(results.simulated_features.mass_ratio * 100).toFixed(2)}%</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Side: Results */}
        <div className="flex-1 bg-slate-900 border border-slate-800 rounded-xl p-8 overflow-y-auto shadow-lg">
          {loading ? (
             <div className="h-full flex flex-col items-center justify-center text-slate-500">
               <BrainCircuit className="w-16 h-16 mb-4 animate-pulse text-cyan-500/50" />
               <p className="animate-pulse">Running Segmentation & Extraction Agents...</p>
             </div>
          ) : results ? (
            <div>
              <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2 border-b border-slate-800 pb-4">
                Similar Retrieved Cases
              </h2>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {results.results.map((r: any, idx: number) => (
                  <div key={r.case_id} className="bg-slate-950 border border-slate-800 rounded-xl overflow-hidden hover:border-cyan-500/50 transition-colors shadow-md relative group">
                    <div className="absolute top-2 left-2 bg-cyan-500 text-slate-950 text-xs font-bold px-2 py-1 rounded shadow-lg z-10">
                      #{idx + 1}
                    </div>
                    <div className="absolute top-2 right-2 bg-slate-900/80 backdrop-blur border border-slate-700 text-cyan-400 text-xs font-bold px-2 py-1 rounded shadow-lg z-10">
                      {(r.similarity * 100).toFixed(1)}% Match
                    </div>
                    <div className="aspect-video bg-black relative">
                      <img src={getOverlayUrl(r.case_id, r.slice_index, [1, 2])} alt="Match" className="absolute inset-0 w-full h-full object-contain group-hover:scale-105 transition-transform duration-500" />
                    </div>
                    <div className="p-4 border-t border-slate-800">
                      <div className="font-bold text-white text-lg">{r.case_id.toUpperCase()}</div>
                      <div className="text-sm text-slate-400 mt-1">Matched at Slice {r.slice_index}</div>
                    </div>
                  </div>
                ))}
              </div>

              {/* LLM Synthesis Panel */}
              <div className="mt-8 pt-8 border-t border-slate-800">
                <h2 className="text-xl font-bold text-white mb-6 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <MessageSquare className="w-6 h-6 text-purple-400" />
                    Decision Agent Synthesis (Llama 3.1 8B)
                  </div>
                  {llmLatency && (
                    <div className="flex items-center gap-1 text-sm text-slate-400 font-normal bg-slate-950 px-3 py-1 rounded-full border border-slate-800">
                      <Clock className="w-4 h-4" /> {(llmLatency / 1000).toFixed(2)}s latency
                    </div>
                  )}
                </h2>
                
                {results.segmentation_warning === "anatomically_inconsistent_prediction" && (
                  <div className="mb-6 bg-red-950/30 border border-red-900/50 p-4 rounded-xl flex items-start gap-3">
                    <AlertTriangle className="w-6 h-6 text-red-500 shrink-0 mt-0.5" />
                    <div>
                      <h4 className="text-red-400 font-bold mb-1">Segmentation Warning: Anatomically Inconsistent Prediction</h4>
                      <p className="text-red-300/80 text-sm">
                        The underlying U-Net segmentation predicted a tumor but detected no surrounding liver tissue (&lt;0.5%). This is highly irregular and suggests the segmentation mask may be a hallucination. Upstream FAISS candidates exhibiting this pattern have been filtered.
                      </p>
                    </div>
                  </div>
                )}

                {llmLoading ? (
                  <div className="bg-slate-950/50 border border-slate-800 rounded-xl p-8 flex flex-col items-center justify-center text-slate-400">
                    <BrainCircuit className="w-10 h-10 mb-4 animate-pulse text-purple-500/50" />
                    <p className="animate-pulse font-medium">Synthesizing multimodal clinical evidence...</p>
                    <p className="text-xs mt-2 opacity-60">Running inference on local Ollama engine</p>
                  </div>
                ) : llmSynthesis ? (
                  <div className="bg-slate-950 border border-slate-800 rounded-xl p-6 space-y-6">
                    <div>
                      <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Clinical Synthesis</h4>
                      <p className="text-slate-200 leading-relaxed text-sm">{llmSynthesis.evidence_synthesis}</p>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Segmentation Findings</h4>
                        <p className="text-slate-300 text-sm">{llmSynthesis.segmentation_findings}</p>
                      </div>
                      <div>
                        <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Retrieval Findings</h4>
                        <p className="text-slate-300 text-sm">{llmSynthesis.retrieval_findings}</p>
                      </div>
                    </div>
                    <div className="pt-4 border-t border-slate-800/50 flex flex-col gap-3">
                      <div className="flex gap-2 text-sm">
                        <span className="font-bold text-slate-400 shrink-0">Note:</span>
                        <span className="text-slate-300">{llmSynthesis.clinical_note}</span>
                      </div>
                      <div className="flex gap-2 text-sm">
                        <span className="font-bold text-slate-400 shrink-0">Limitations:</span>
                        <span className="text-yellow-500/80">{llmSynthesis.limitations}</span>
                      </div>
                    </div>
                  </div>
                ) : null}
              </div>
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-slate-500">
              <UploadCloud className="w-16 h-16 mb-4 opacity-20" />
              <p>Upload a CT slice to initiate the retrieval pipeline.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
