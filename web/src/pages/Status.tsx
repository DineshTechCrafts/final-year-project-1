import { useEffect, useState } from 'react';
import { fetchAPI } from '../api/client';
import { Server, ShieldAlert } from 'lucide-react';

export default function Status() {
  const [status, setStatus] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAPI('/model/status')
      .then(setStatus)
      .catch(err => setError(err.message));
  }, []);

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-white flex items-center gap-3 mb-8">
        <Server className="w-8 h-8 text-cyan-400" />
        System Status
      </h1>

      {error ? (
        <div className="bg-red-950/50 border border-red-900 p-6 rounded-xl flex items-center gap-4 text-red-400">
          <ShieldAlert className="w-8 h-8" />
          <div>
            <h3 className="font-bold text-lg text-red-300">Backend Unavailable</h3>
            <p className="text-sm">{error}</p>
          </div>
        </div>
      ) : status ? (
        <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-xl">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-950 border-b border-slate-800 text-slate-400 uppercase text-xs tracking-wider">
              <tr>
                <th className="px-6 py-4 font-medium">Component</th>
                <th className="px-6 py-4 font-medium">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {Object.entries(status).map(([key, val]: [string, any]) => (
                <tr key={key} className="hover:bg-slate-800/30">
                  <td className="px-6 py-4 font-medium text-white">{key}</td>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-2">
                      <div className={`w-2.5 h-2.5 rounded-full ${
                        val.color === 'green' ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]' :
                        val.color === 'blue' ? 'bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.6)]' :
                        val.color === 'yellow' ? 'bg-yellow-500 shadow-[0_0_8px_rgba(234,179,8,0.6)]' :
                        'bg-slate-500'
                      }`}></div>
                      <span className={`font-semibold ${
                        val.color === 'green' ? 'text-green-400' :
                        val.color === 'blue' ? 'text-blue-400' :
                        val.color === 'yellow' ? 'text-yellow-400' :
                        'text-slate-400'
                      }`}>{val.status}</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="text-slate-400 text-center py-12">Loading status...</div>
      )}
    </div>
  );
}
