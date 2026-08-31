import { useEffect, useState } from 'react';
import { fetchAPI, getImageUrl, getOverlayUrl, getMaskUrl } from '../api/client';
import { ChevronLeft, ChevronRight } from 'lucide-react';

export default function Analysis() {
  const [patients, setPatients] = useState<any[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<string>("hcc_055");
  const [sliceIndex, setSliceIndex] = useState(35);
  const [totalSlices, setTotalSlices] = useState(71);
  const [viewMode, setViewMode] = useState<'original' | 'mask' | 'overlay'>('overlay');
  
  const [classes, setClasses] = useState({
    1: { name: 'Liver', active: true, color: 'text-red-400' },
    2: { name: 'Mass', active: true, color: 'text-yellow-400' },
    3: { name: 'Portal Vein', active: true, color: 'text-blue-400' },
    4: { name: 'Abdominal Aorta', active: true, color: 'text-green-400' }
  });

  useEffect(() => {
    fetchAPI('/patients').then(res => {
      setPatients(res.patients);
      if (res.patients.length > 0 && !res.patients.find((p: any) => p.patient_id === "hcc_055")) {
        setSelectedPatient(res.patients[0].patient_id);
      }
    }).catch(console.error);
  }, []);

  useEffect(() => {
    if (selectedPatient) {
      fetchAPI(`/patients/${selectedPatient}/slices`).then(res => {
        setTotalSlices(res.total_slices);
        setSliceIndex(Math.floor(res.total_slices / 2));
      }).catch(console.error);
    }
  }, [selectedPatient]);

  const activeClasses = Object.entries(classes)
    .filter(([_, c]) => c.active)
    .map(([id]) => parseInt(id));

  const toggleClass = (id: number) => {
    setClasses(prev => ({
      ...prev,
      [id]: { ...prev[id as keyof typeof prev], active: !prev[id as keyof typeof prev].active }
    }));
  };

  return (
    <div className="flex h-full">
      {/* Left Sidebar - Controls */}
      <div className="w-80 border-r border-slate-800 bg-slate-900/50 p-6 overflow-y-auto">
        <h2 className="text-lg font-bold mb-6 text-white">CT Analysis</h2>
        
        <div className="mb-8">
          <label className="block text-xs font-medium text-slate-400 mb-2 uppercase tracking-wider">Select Patient</label>
          <select 
            className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-sm focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none"
            value={selectedPatient}
            onChange={(e) => setSelectedPatient(e.target.value)}
          >
            {patients.map(p => (
              <option key={p.patient_id} value={p.patient_id}>{p.patient_id.toUpperCase()} ({p.split})</option>
            ))}
          </select>
        </div>

        <div className="mb-8">
          <label className="block text-xs font-medium text-slate-400 mb-3 uppercase tracking-wider">View Mode</label>
          <div className="flex bg-slate-950 rounded-lg p-1 border border-slate-800">
            {['original', 'mask', 'overlay'].map(mode => (
              <button
                key={mode}
                className={`flex-1 text-xs py-1.5 rounded-md capitalize transition-colors ${viewMode === mode ? 'bg-slate-800 text-white' : 'text-slate-400 hover:text-white'}`}
                onClick={() => setViewMode(mode as any)}
              >
                {mode}
              </button>
            ))}
          </div>
        </div>

        <div className="mb-8">
          <label className="block text-xs font-medium text-slate-400 mb-3 uppercase tracking-wider">Segmentation Classes</label>
          <div className="space-y-2">
            {Object.entries(classes).map(([id, c]) => (
              <label key={id} className="flex items-center gap-3 p-2.5 rounded-lg border border-slate-800 bg-slate-950 cursor-pointer hover:border-slate-700 transition-colors">
                <input 
                  type="checkbox" 
                  checked={c.active} 
                  onChange={() => toggleClass(parseInt(id))}
                  className="w-4 h-4 rounded border-slate-700 text-cyan-500 focus:ring-cyan-500 focus:ring-offset-slate-950 bg-slate-900"
                />
                <span className={`text-sm font-medium ${c.color}`}>{c.name}</span>
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* Main Viewer Area */}
      <div className="flex-1 flex flex-col bg-slate-950 p-6 items-center justify-center relative">
        <div className="w-full max-w-3xl">
          <div className="flex justify-between items-end mb-4 px-2">
            <div>
              <h3 className="text-xl font-bold text-white">{selectedPatient.toUpperCase()}</h3>
              <p className="text-sm text-slate-400">Slice {sliceIndex} / {totalSlices - 1}</p>
            </div>
            <div className="flex gap-2">
              <button 
                onClick={() => setSliceIndex(Math.max(0, sliceIndex - 1))}
                className="p-2 rounded-lg bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-300 transition-colors"
                disabled={sliceIndex <= 0}
              >
                <ChevronLeft className="w-5 h-5" />
              </button>
              <button 
                onClick={() => setSliceIndex(Math.min(totalSlices - 1, sliceIndex + 1))}
                className="p-2 rounded-lg bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-300 transition-colors"
                disabled={sliceIndex >= totalSlices - 1}
              >
                <ChevronRight className="w-5 h-5" />
              </button>
            </div>
          </div>

          <div className="relative aspect-square w-full bg-black rounded-xl border border-slate-800 overflow-hidden shadow-2xl flex items-center justify-center">
            {/* Base Image */}
            {(viewMode === 'original' || viewMode === 'overlay') && (
              <img 
                src={getImageUrl(selectedPatient, sliceIndex)} 
                alt="CT Slice" 
                className="absolute inset-0 w-full h-full object-contain"
              />
            )}
            
            {/* Mask/Overlay */}
            {(viewMode === 'mask' || viewMode === 'overlay') && (
              <img 
                src={viewMode === 'mask' 
                  ? getMaskUrl(selectedPatient, sliceIndex)
                  : getOverlayUrl(selectedPatient, sliceIndex, activeClasses)} 
                alt="Segmentation Mask" 
                className={`absolute inset-0 w-full h-full object-contain ${viewMode === 'overlay' ? 'opacity-80' : ''} ${viewMode === 'mask' ? 'brightness-200' : ''}`}
                style={{ mixBlendMode: viewMode === 'mask' ? 'normal' : 'normal' }}
              />
            )}
          </div>

          <div className="mt-8 px-2">
            <input 
              type="range" 
              min={0} 
              max={totalSlices - 1} 
              value={sliceIndex} 
              onChange={(e) => setSliceIndex(parseInt(e.target.value))}
              className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
