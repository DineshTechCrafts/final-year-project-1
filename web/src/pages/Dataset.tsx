import { useEffect, useState } from 'react';
import { fetchAPI } from '../api/client';
import { Database, Users, Layers, ShieldCheck, Activity } from 'lucide-react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';

export default function Dataset() {
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    fetchAPI('/stats').then(setStats).catch(console.error);
  }, []);

  if (!stats) return <div className="p-8 text-slate-400">Loading dataset metadata...</div>;

  const totalSlices = stats.split_distribution.total || 11277;
  
  const pieData = [
    { name: 'Train', value: stats.split_distribution.train, color: '#22d3ee' },
    { name: 'Validation', value: stats.split_distribution.validation, color: '#3b82f6' },
    { name: 'Test', value: stats.split_distribution.test, color: '#a855f7' },
  ];

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold text-white flex items-center gap-3 mb-8">
        <Database className="w-8 h-8 text-cyan-400" />
        Dataset Overview
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg">
          <h2 className="text-xl font-bold text-white mb-6 border-b border-slate-800 pb-3">Summary</h2>
          
          <div className="space-y-4">
            <div className="flex justify-between items-center bg-slate-950 p-4 rounded-lg border border-slate-800">
              <div className="flex items-center gap-3">
                <Users className="w-5 h-5 text-cyan-400" />
                <span className="text-slate-300 font-medium">Processed Patients</span>
              </div>
              <span className="text-xl font-bold text-white">{stats.total_patients}</span>
            </div>
            
            <div className="flex justify-between items-center bg-slate-950 p-4 rounded-lg border border-slate-800">
              <div className="flex items-center gap-3">
                <Layers className="w-5 h-5 text-blue-400" />
                <span className="text-slate-300 font-medium">Processed Slices</span>
              </div>
              <span className="text-xl font-bold text-white">{totalSlices.toLocaleString()}</span>
            </div>

            <div className="flex justify-between items-center bg-slate-950 p-4 rounded-lg border border-slate-800">
              <div className="flex items-center gap-3">
                <ShieldCheck className="w-5 h-5 text-green-400" />
                <span className="text-slate-300 font-medium">Patient Leakage</span>
              </div>
              <span className="text-sm font-bold text-green-400 bg-green-950/50 px-2 py-1 rounded border border-green-900">0 Overlap</span>
            </div>

            <div className="flex justify-between items-center bg-slate-950 p-4 rounded-lg border border-slate-800">
              <div className="flex items-center gap-3">
                <Activity className="w-5 h-5 text-purple-400" />
                <span className="text-slate-300 font-medium">Skipped Patients</span>
              </div>
              <span className="text-xl font-bold text-slate-500">1 (HCC_048)</span>
            </div>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg">
          <h2 className="text-xl font-bold text-white mb-6 border-b border-slate-800 pb-3">Split Distribution</h2>
          
          <div className="mb-6 space-y-3">
            <div className="flex justify-between items-center text-sm">
              <span className="text-slate-400">Train (72 patients)</span>
              <span className="text-cyan-400 font-mono font-bold">{stats.split_distribution.train?.toLocaleString()} slices</span>
            </div>
            <div className="flex justify-between items-center text-sm">
              <span className="text-slate-400">Validation (15 patients)</span>
              <span className="text-blue-400 font-mono font-bold">{stats.split_distribution.validation?.toLocaleString()} slices</span>
            </div>
            <div className="flex justify-between items-center text-sm">
              <span className="text-slate-400">Test (17 patients)</span>
              <span className="text-purple-400 font-mono font-bold">{stats.split_distribution.test?.toLocaleString()} slices</span>
            </div>
          </div>

          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                  stroke="none"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{backgroundColor: '#0f172a', borderColor: '#1e293b'}}
                  itemStyle={{color: '#fff'}}
                  formatter={(value: any) => [`${value.toLocaleString()} slices`, 'Count']}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-xl font-bold text-white mb-6 border-b border-slate-800 pb-3">Semantic Segmentation Classes</h2>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 flex flex-col items-center justify-center text-center">
            <span className="text-3xl font-bold text-slate-600 mb-1">0</span>
            <span className="text-xs text-slate-400 uppercase tracking-wider">Background</span>
          </div>
          <div className="bg-slate-950 p-4 rounded-lg border border-red-900/30 flex flex-col items-center justify-center text-center">
            <span className="text-3xl font-bold text-red-400 mb-1">1</span>
            <span className="text-xs text-slate-400 uppercase tracking-wider">Liver</span>
          </div>
          <div className="bg-slate-950 p-4 rounded-lg border border-yellow-900/30 flex flex-col items-center justify-center text-center">
            <span className="text-3xl font-bold text-yellow-400 mb-1">2</span>
            <span className="text-xs text-slate-400 uppercase tracking-wider">Mass</span>
          </div>
          <div className="bg-slate-950 p-4 rounded-lg border border-blue-900/30 flex flex-col items-center justify-center text-center">
            <span className="text-3xl font-bold text-blue-400 mb-1">3</span>
            <span className="text-xs text-slate-400 uppercase tracking-wider">Portal Vein</span>
          </div>
          <div className="bg-slate-950 p-4 rounded-lg border border-green-900/30 flex flex-col items-center justify-center text-center">
            <span className="text-3xl font-bold text-green-400 mb-1">4</span>
            <span className="text-xs text-slate-400 uppercase tracking-wider">Abdominal Aorta</span>
          </div>
        </div>
      </div>

    </div>
  );
}
