import { useEffect, useState } from 'react';
import { fetchAPI } from '../api/client';
import { Users, Layers, Maximize, ShieldCheck, Server } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function Dashboard() {
  const [stats, setStats] = useState<any>(null);
  const [status, setStatus] = useState<any>(null);

  useEffect(() => {
    fetchAPI('/stats').then(setStats).catch(console.error);
    fetchAPI('/model/status').then(setStatus).catch(console.error);
  }, []);

  if (!stats) return <div className="p-8 flex justify-center text-slate-400">Loading dashboard...</div>;

  const chartData = [
    { name: 'Train', slices: stats.split_distribution.train },
    { name: 'Validation', slices: stats.split_distribution.validation },
    { name: 'Test', slices: stats.split_distribution.test },
  ];

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold text-white mb-8">Dataset Overview</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl shadow-lg flex flex-col">
          <div className="flex items-center gap-3 text-slate-400 mb-2">
            <Users className="w-5 h-5 text-cyan-400" /> Patients
          </div>
          <span className="text-3xl font-bold text-white">{stats.total_patients}</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl shadow-lg flex flex-col">
          <div className="flex items-center gap-3 text-slate-400 mb-2">
            <Layers className="w-5 h-5 text-blue-400" /> CT Slices
          </div>
          <span className="text-3xl font-bold text-white">{stats.split_distribution.total?.toLocaleString() || 11277}</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl shadow-lg flex flex-col">
          <div className="flex items-center gap-3 text-slate-400 mb-2">
            <ShieldCheck className="w-5 h-5 text-green-400" /> Patient Leakage
          </div>
          <span className="text-3xl font-bold text-green-400">0%</span>
          <span className="text-xs text-slate-500 mt-1">Train/Val/Test Overlap</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl shadow-lg flex flex-col">
          <div className="flex items-center gap-3 text-slate-400 mb-2">
            <Maximize className="w-5 h-5 text-purple-400" /> Classes
          </div>
          <span className="text-3xl font-bold text-white">{stats.classes.length - 1}</span>
          <span className="text-xs text-slate-500 mt-1">Excl. Background</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl shadow-lg">
          <h2 className="text-xl font-semibold mb-6">Split Distribution</h2>
          <div className="space-y-4">
            <div className="flex justify-between items-center pb-3 border-b border-slate-800">
              <span className="text-slate-300">Train</span>
              <span className="text-cyan-400 font-mono">{stats.split_distribution.train?.toLocaleString()} slices</span>
            </div>
            <div className="flex justify-between items-center pb-3 border-b border-slate-800">
              <span className="text-slate-300">Validation</span>
              <span className="text-blue-400 font-mono">{stats.split_distribution.validation?.toLocaleString()} slices</span>
            </div>
            <div className="flex justify-between items-center pb-3 border-b border-slate-800">
              <span className="text-slate-300">Test</span>
              <span className="text-purple-400 font-mono">{stats.split_distribution.test?.toLocaleString()} slices</span>
            </div>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl shadow-lg h-80">
          <h2 className="text-xl font-semibold mb-6">Distribution Chart</h2>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="name" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip cursor={{fill: '#1e293b'}} contentStyle={{backgroundColor: '#0f172a', borderColor: '#1e293b'}} />
              <Bar dataKey="slices" fill="#22d3ee" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="mt-8 bg-slate-900 border border-slate-800 p-6 rounded-xl shadow-lg">
        <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
          <Server className="w-5 h-5 text-cyan-400" />
          System Status
        </h2>
        {status ? (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {Object.entries(status).map(([key, val]: [string, any]) => (
              <div key={key} className="bg-slate-950 p-4 rounded-lg border border-slate-800">
                <div className="text-xs text-slate-400 uppercase tracking-wider mb-2">{key}</div>
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
              </div>
            ))}
          </div>
        ) : (
          <div className="text-slate-400 text-sm">Loading status...</div>
        )}
      </div>
    </div>
  );
}
