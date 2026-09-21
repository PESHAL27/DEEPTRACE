import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/layout/Sidebar';
import type { NavRoute } from './components/layout/Sidebar';
import { TopHeader } from './components/layout/TopHeader';
import { HomePage } from './components/home/HomePage';
import { NewAnalysisPage } from './components/analysis/NewAnalysisPage';
import { AnalysisResultsPage } from './components/analysis/AnalysisResultsPage';
import { HistoryPage } from './components/history/HistoryPage';
import { ReportsPage } from './components/reports/ReportsPage';
import { ModelsPage } from './components/models/ModelsPage';
import { BenchmarksPage } from './components/benchmarks/BenchmarksPage';
import { SettingsPage } from './components/settings/SettingsPage';
import type { Analysis, ForensicReport, ModelInfo } from './types/forensics';
import { forensicApiService } from './services/api';

export const App: React.FC = () => {
  const [currentRoute, setCurrentRoute] = useState<NavRoute>('home');
  const [activeAnalysis, setActiveAnalysis] = useState<Analysis | null>(null);

  // Data states
  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [reports, setReports] = useState<ForensicReport[]>([]);

  // Load initial data from api service
  useEffect(() => {
    const loadData = async () => {
      try {
        const [historyData, modelsData, reportsData] = await Promise.all([
          forensicApiService.getAnalysisHistory(),
          forensicApiService.getModels(),
          forensicApiService.getReports(),
        ]);
        setAnalyses(historyData);
        setModels(modelsData);
        setReports(reportsData);
        if (historyData && historyData.length > 0) {
          setActiveAnalysis(historyData[0]);
        }
      } catch (err) {
        console.error('Error loading initial data:', err);
      }
    };
    loadData();
  }, []);

  const handleSelectAnalysis = (analysis: Analysis) => {
    setActiveAnalysis(analysis);
    setCurrentRoute('results');
  };

  const handleAnalysisComplete = (newAnalysis: Analysis) => {
    setAnalyses((prev) => [newAnalysis, ...prev]);
    setActiveAnalysis(newAnalysis);
    forensicApiService.getReports().then(setReports).catch(() => {});
    setCurrentRoute('results');
  };

  const handleSearch = (query: string) => {
    if (query.trim().length > 0 && currentRoute !== 'history') {
      setCurrentRoute('history');
    }
  };

  return (
    <div className="app-shell">
      {/* Persistent Left Sidebar (~170px) */}
      <Sidebar 
        currentRoute={currentRoute} 
        onNavigate={(route) => setCurrentRoute(route)} 
      />

      {/* Main Content Workspace */}
      <div className="app-main">
        {/* Top Header (~58px) */}
        <TopHeader onSearch={handleSearch} />

        {/* Page Content View */}
        <main className="content-container">
          {currentRoute === 'home' && (
            <HomePage
              recentAnalyses={analyses}
              onNavigateToUpload={() => setCurrentRoute('analyze')}
              onNavigateToDemo={() => {
                if (analyses.length > 0) {
                  setActiveAnalysis(analyses[0]);
                  setCurrentRoute('results');
                } else {
                  setCurrentRoute('analyze');
                }
              }}
              onSelectAnalysis={handleSelectAnalysis}
              onNavigateToHistory={() => setCurrentRoute('history')}
              onNavigateToReports={() => setCurrentRoute('reports')}
            />
          )}

          {currentRoute === 'analyze' && (
            <NewAnalysisPage
              onAnalysisComplete={handleAnalysisComplete}
              onCancel={() => setCurrentRoute('home')}
            />
          )}

          {currentRoute === 'results' && (
            <AnalysisResultsPage
              analysis={activeAnalysis}
              onBack={() => setCurrentRoute('home')}
            />
          )}

          {currentRoute === 'history' && (
            <HistoryPage
              analyses={analyses}
              onSelectAnalysis={handleSelectAnalysis}
            />
          )}

          {currentRoute === 'reports' && (
            <ReportsPage reports={reports} />
          )}

          {currentRoute === 'models' && (
            <ModelsPage models={models} />
          )}

          {currentRoute === 'benchmarks' && (
            <BenchmarksPage />
          )}

          {currentRoute === 'settings' && (
            <SettingsPage />
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
