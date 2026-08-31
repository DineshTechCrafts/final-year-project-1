
import { Workflow, User, Image as ImageIcon, Scan, Brain, Search, CheckCircle } from 'lucide-react';

export default function Agents() {
  return (
    <div className="p-8 max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold text-white flex items-center gap-3 mb-12">
        <Workflow className="w-8 h-8 text-cyan-400" />
        Multi-Agent Architecture
      </h1>

      <div className="relative flex flex-col items-center max-w-2xl mx-auto space-y-6">
        
        {/* Connection Line */}
        <div className="absolute top-0 bottom-0 left-1/2 w-1 bg-slate-800 -translate-x-1/2 z-0"></div>

        {/* User Node */}
        <div className="relative z-10 w-full max-w-sm bg-slate-900 border-2 border-slate-700 rounded-xl p-4 flex items-center gap-4 shadow-lg">
          <div className="p-3 bg-slate-800 rounded-lg text-slate-300">
            <User className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-bold text-white">Clinical User</h3>
            <p className="text-xs text-slate-400">Uploads CT Scan for analysis</p>
          </div>
        </div>

        {/* Processing Node */}
        <div className="relative z-10 w-full max-w-sm bg-slate-900 border-2 border-cyan-900 rounded-xl p-4 flex items-center gap-4 shadow-lg shadow-cyan-900/20">
          <div className="p-3 bg-cyan-950 rounded-lg text-cyan-400">
            <ImageIcon className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-bold text-white">Image Processing Agent</h3>
            <p className="text-xs text-slate-400">DICOM to PNG, Windowing (HU config)</p>
          </div>
          <span className="ml-auto px-2 py-1 bg-green-950 text-green-400 text-[10px] rounded border border-green-900">Implemented</span>
        </div>

        {/* Segmentation Node */}
        <div className="relative z-10 w-full max-w-sm bg-slate-900 border-2 border-blue-900 rounded-xl p-4 flex items-center gap-4 shadow-lg shadow-blue-900/20">
          <div className="p-3 bg-blue-950 rounded-lg text-blue-400">
            <Scan className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-bold text-white">Segmentation Agent</h3>
            <p className="text-xs text-slate-400">U-Net based organ/tumor masking</p>
          </div>
          <span className="ml-auto px-2 py-1 bg-green-950 text-green-400 text-[10px] rounded border border-green-900">Implemented</span>
        </div>

        {/* Feature Extraction Node */}
        <div className="relative z-10 w-full max-w-sm bg-slate-900 border-2 border-purple-900 rounded-xl p-4 flex items-center gap-4 shadow-lg shadow-purple-900/20">
          <div className="p-3 bg-purple-950 rounded-lg text-purple-400">
            <Brain className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-bold text-white">Feature Extraction Agent</h3>
            <p className="text-xs text-slate-400">ResNet Visual / Structural Embedding</p>
          </div>
          <span className="ml-auto px-2 py-1 bg-yellow-950 text-yellow-500 text-[10px] rounded border border-yellow-900">Prototype</span>
        </div>

        {/* Retrieval Node */}
        <div className="relative z-10 w-full max-w-sm bg-slate-900 border-2 border-orange-900 rounded-xl p-4 flex items-center gap-4 shadow-lg shadow-orange-900/20">
          <div className="p-3 bg-orange-950 rounded-lg text-orange-400">
            <Search className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-bold text-white">Retrieval Agent</h3>
            <p className="text-xs text-slate-400">FAISS similarity & ranking</p>
          </div>
          <span className="ml-auto px-2 py-1 bg-yellow-950 text-yellow-500 text-[10px] rounded border border-yellow-900">Demo Mode</span>
        </div>

        {/* Results Node */}
        <div className="relative z-10 w-full max-w-sm bg-slate-900 border-2 border-green-900 rounded-xl p-4 flex items-center gap-4 shadow-lg shadow-green-900/20">
          <div className="p-3 bg-green-950 rounded-lg text-green-400">
            <CheckCircle className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-bold text-white">Decision / Results Agent</h3>
            <p className="text-xs text-slate-400">Explanation & evidence synthesis</p>
          </div>
          <span className="ml-auto px-2 py-1 bg-slate-800 text-slate-400 text-[10px] rounded border border-slate-700">Planned</span>
        </div>

      </div>
    </div>
  );
}
