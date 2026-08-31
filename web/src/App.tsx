
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Activity, Database, LayoutDashboard, Brain, TestTube2, Workflow } from 'lucide-react';
import Dashboard from './pages/Dashboard';
import Patients from './pages/Patients';
import Analysis from './pages/Analysis';
import Retrieval from './pages/Retrieval';
import Upload from './pages/Upload';
import Agents from './pages/Agents';
import Status from './pages/Status';
import Dataset from './pages/Dataset';

const Sidebar = () => {
  const location = useLocation();
  const links = [
    { name: 'Dashboard', icon: LayoutDashboard, path: '/' },
    { name: 'Upload Scan', icon: Activity, path: '/upload' },
    { name: 'CT Analysis', icon: Activity, path: '/analysis' },
    { name: 'Case Retrieval', icon: Brain, path: '/retrieval' },
    { name: 'Patients', icon: Database, path: '/patients' },
    { name: 'Dataset', icon: Database, path: '/dataset' },
    { name: 'Multi-Agent System', icon: Workflow, path: '/agents' },
    { name: 'System Status', icon: TestTube2, path: '/status' },
  ];

  return (
    <div className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col h-screen">
      <div className="p-6 border-b border-slate-800">
        <h1 className="text-xl font-bold text-cyan-400 flex items-center gap-2">
          <Brain className="w-6 h-6" />
          Liver Cancer AI
        </h1>
        <p className="text-xs text-slate-400 mt-2">Advanced Medical Imaging Retrieval</p>
      </div>
      <div className="flex-1 py-4">
        <nav className="space-y-1 px-3">
          {links.map((link) => {
            const active = location.pathname === link.path || (link.path !== '/' && location.pathname.startsWith(link.path));
            return (
              <Link
                key={link.name}
                to={link.path}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  active ? 'bg-cyan-950/50 text-cyan-400' : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                }`}
              >
                <link.icon className="w-5 h-5" />
                {link.name}
              </Link>
            );
          })}
        </nav>
      </div>
      <div className="p-4 border-t border-slate-800">
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <div className="w-2 h-2 rounded-full bg-green-500"></div>
          Backend Connected
        </div>
        <div className="mt-4 text-[10px] text-slate-500">
          Research prototype — not intended for clinical diagnosis or treatment decisions.
        </div>
      </div>
    </div>
  );
};

function App() {
  return (
    <BrowserRouter>
      <div className="flex h-screen bg-slate-950 text-slate-50 overflow-hidden font-sans">
        <Sidebar />
        <div className="flex-1 overflow-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/patients" element={<Patients />} />
            <Route path="/analysis" element={<Analysis />} />
            <Route path="/retrieval" element={<Retrieval />} />
            <Route path="/agents" element={<Agents />} />
            <Route path="/status" element={<Status />} />
            <Route path="/dataset" element={<Dataset />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  );
}

export default App;
