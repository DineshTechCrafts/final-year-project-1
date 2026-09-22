import { useEffect, useState } from 'react';
import { fetchAPI, getOverlayUrl } from '../api/client';
import { Search, BrainCircuit, ScanSearch, Info } from 'lucide-react';

export default function Retrieval() {
  const [patients, setPatients] = useState<any[]>([]);
  const [queryPatient, setQueryPatient] = useState("");
  const [querySlice, setQuerySlice] = useState(35);
  const [totalSlices, setTotalSlices] = useState(71);
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchAPI('/patients').then(res => {
      setPatients(res.patients);
      if (res.patients.length > 0) {
        setQueryPatient(res.patients[0].patient_id);
      }
    }).catch(console.error);
  }, []);

  const handleSearch = async () => {
    setLoading(true);
    try {
      const data = await fetchAPI(`/retrieval/similar/${queryPatient}?slice_index=${querySlice}`);
      setResults(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 max-w-7xl mx-auto flex flex-col h-full">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-white flex items-center gap-3">
          <ScanSearch className="w-8 h-8 text-cyan-400" />
          Similar Case Retrieval
        </h1>
        {results?.demo_mode && (
          <div className="flex items-center gap-2 px-4 py-2 bg-yellow-950 border border-yellow-900 rounded-lg text-yellow-500 text-sm font-medium">
            <Info className="w-4 h-4" />
            Demo Retrieval Mode
          </div>
        )}
      </div>

      <div className="flex gap-8 flex-1 min-h-0">
        {/* Left Side: Query */}
        <div className="w-80 flex flex-col gap-6">
          <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl shadow-lg">
            <h2 className="text-lg font-semibold text-white mb-4">Query Parameters</h2>
            
            <label className="block text-xs font-medium text-slate-400 mb-2 uppercase tracking-wider">Target Patient</label>
            <select 
              className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-sm focus:border-cyan-500 focus:ring-1 outline-none mb-4"
              value={queryPatient}
              onChange={(e) => setQueryPatient(e.target.value)}
            >
              {patients.map(p => (
                <option key={p.patient_id} value={p.patient_id}>{p.patient_id.toUpperCase()}</option>
              ))}
            </select>
            
            <label className="block text-xs font-medium text-slate-400 mb-2 uppercase tracking-wider">Query Slice Index</label>
            <input 
              type="number" 
              value={querySlice}
              onChange={(e) => setQuerySlice(parseInt(e.target.value))}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-sm focus:border-cyan-500 focus:ring-1 outline-none mb-6"
            />
            
            <button 
              onClick={handleSearch}
              disabled={loading}
              className="w-full flex justify-center items-center gap-2 bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-3 px-4 rounded-lg transition-colors disabled:opacity-50"
            >
              {loading ? <BrainCircuit className="w-5 h-5 animate-pulse" /> : <Search className="w-5 h-5" />}
              {loading ? 'Extracting Features...' : 'Find Similar Cases'}
            </button>
          </div>
          
          {queryPatient && (
            <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl shadow-lg flex-1">
              <h3 className="text-sm font-medium text-slate-400 mb-3 uppercase tracking-wider">Query Preview</h3>
              <div className="aspect-square bg-black rounded-lg border border-slate-800 relative overflow-hidden">
                <img src={getOverlayUrl(queryPatient, querySlice, [1, 2])} alt="Query" className="absolute inset-0 w-full h-full object-contain" />
              </div>
            </div>
          )}
        </div>

        {/* Right Side: Results */}
        <div className="flex-1 bg-slate-900 border border-slate-800 rounded-xl p-8 overflow-y-auto shadow-lg">
          {results ? (
            <div>
              <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                Top Retrieved Cases
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {results.results.map((r: any, idx: number) => (
                  <div key={r.candidate_case_id} className="bg-slate-950 border border-slate-800 rounded-xl overflow-hidden hover:border-cyan-500/50 transition-colors shadow-md relative group">
                    <div className="absolute top-2 left-2 bg-cyan-500 text-slate-950 text-xs font-bold px-2 py-1 rounded shadow-lg z-10">
                      #{idx + 1}
                    </div>
                    <div className="absolute top-2 right-2 bg-slate-900/80 backdrop-blur border border-slate-700 text-cyan-400 text-xs font-bold px-2 py-1 rounded shadow-lg z-10">
                      {(r.similarity * 100).toFixed(1)}% Match
                    </div>
                    <div className="aspect-square bg-black relative">
                      <img src={getOverlayUrl(r.candidate_case_id, r.matched_slice, [1, 2])} alt="Match" className="absolute inset-0 w-full h-full object-contain group-hover:scale-105 transition-transform duration-500" />
                    </div>
                    <div className="p-4 border-t border-slate-800">
                      <div className="font-bold text-white text-lg">{r.candidate_case_id.toUpperCase()}</div>
                      <div className="text-sm text-slate-400 mt-1">Matched at Slice {r.matched_slice}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-slate-500">
              <BrainCircuit className="w-16 h-16 mb-4 opacity-20" />
              <p>Select a query slice and click 'Find Similar Cases' to execute retrieval.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
