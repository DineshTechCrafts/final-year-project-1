import { useEffect, useState } from 'react';
import { fetchAPI } from '../api/client';
import { Link } from 'react-router-dom';
import { Database, Search, ArrowRight } from 'lucide-react';

export default function Patients() {
  const [patients, setPatients] = useState<any[]>([]);
  const [search, setSearch] = useState('');

  useEffect(() => {
    fetchAPI('/patients').then(res => setPatients(res.patients)).catch(console.error);
  }, []);

  const filtered = patients.filter(p => p.patient_id.toLowerCase().includes(search.toLowerCase()));

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold text-white flex items-center gap-3">
          <Database className="w-8 h-8 text-cyan-400" />
          Patient Dataset
        </h1>
        <div className="relative">
          <Search className="w-5 h-5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input 
            type="text" 
            placeholder="Search patient ID..." 
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="pl-10 pr-4 py-2 bg-slate-900 border border-slate-800 rounded-lg text-sm text-white focus:ring-1 focus:ring-cyan-500 outline-none w-64"
          />
        </div>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-xl">
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-950/50 border-b border-slate-800 text-slate-400 uppercase text-xs tracking-wider">
            <tr>
              <th className="px-6 py-4 font-medium">Patient ID</th>
              <th className="px-6 py-4 font-medium">Split Partition</th>
              <th className="px-6 py-4 font-medium">CT Slices</th>
              <th className="px-6 py-4 font-medium">Classes Available</th>
              <th className="px-6 py-4 font-medium text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {filtered.map(p => (
              <tr key={p.patient_id} className="hover:bg-slate-800/50 transition-colors">
                <td className="px-6 py-4 font-medium text-white">{p.patient_id.toUpperCase()}</td>
                <td className="px-6 py-4">
                  <span className={`px-2.5 py-1 rounded-full text-[10px] font-bold tracking-wide uppercase ${
                    p.split === 'train' ? 'bg-cyan-950 text-cyan-400 border border-cyan-900' :
                    p.split === 'validation' ? 'bg-blue-950 text-blue-400 border border-blue-900' :
                    'bg-purple-950 text-purple-400 border border-purple-900'
                  }`}>
                    {p.split}
                  </span>
                </td>
                <td className="px-6 py-4 text-slate-300 font-mono">{p.num_slices}</td>
                <td className="px-6 py-4 text-slate-400 text-xs">
                  {p.classes_present.length - 1} semantic classes
                </td>
                <td className="px-6 py-4 text-right">
                  <Link 
                    to={`/analysis`}
                    className="inline-flex items-center gap-1 text-cyan-400 hover:text-cyan-300 font-medium text-xs uppercase tracking-wider"
                  >
                    Analyze <ArrowRight className="w-4 h-4" />
                  </Link>
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={5} className="px-6 py-8 text-center text-slate-500">
                  No patients found matching "{search}"
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
