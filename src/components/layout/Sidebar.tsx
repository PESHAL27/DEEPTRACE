import React from 'react';
import { 
  Home, 
  PlusCircle, 
  Clock, 
  Cpu, 
  FileText, 
  Settings, 
  ShieldCheck,
  BarChart3
} from 'lucide-react';

export type NavRoute = 'home' | 'analyze' | 'history' | 'models' | 'reports' | 'benchmarks' | 'settings' | 'results';

interface SidebarProps {
  currentRoute: NavRoute;
  onNavigate: (route: NavRoute) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ currentRoute, onNavigate }) => {
  const navItems = [
    { id: 'home', label: 'Home', icon: Home },
    { id: 'analyze', label: 'New Analysis', icon: PlusCircle },
    { id: 'history', label: 'Analysis History', icon: Clock },
    { id: 'models', label: 'Models', icon: Cpu },
    { id: 'reports', label: 'Reports', icon: FileText },
    { id: 'benchmarks', label: 'Benchmarks', icon: BarChart3 },
    { id: 'settings', label: 'Settings', icon: Settings },
  ] as const;

  return (
    <aside className="app-sidebar">
      <div className="sidebar-top">
        {/* Brand Header */}
        <div className="sidebar-brand" onClick={() => onNavigate('home')}>
          <div className="brand-icon-wrapper">
            <ShieldCheck size={26} strokeWidth={2.2} />
          </div>
          <div className="brand-title-box">
            <span className="brand-name">DEEPTRACE</span>
            <span className="brand-subtitle">MEDIA FORENSICS</span>
          </div>
        </div>

        {/* Navigation Menu */}
        <nav className="sidebar-nav">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentRoute === item.id || (item.id === 'analyze' && currentRoute === 'results');
            return (
              <button
                key={item.id}
                className={`nav-item ${isActive ? 'active' : ''}`}
                onClick={() => onNavigate(item.id)}
                title={item.label}
              >
                <span className="nav-icon">
                  <Icon size={16} strokeWidth={isActive ? 2.2 : 1.8} />
                </span>
                <span className="nav-text">{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Bottom Trust Panel */}
      <div className="sidebar-bottom">
        <div className="trust-panel">
          <div className="trust-title">
            Trusted Media.<br />Safer Tomorrow.
          </div>
          <div className="trust-desc">
            Detect. Analyze. Verify.
          </div>
        </div>
      </div>
    </aside>
  );
};
