import React, { useState } from 'react';
import { Search, Bell, ChevronDown } from 'lucide-react';

interface TopHeaderProps {
  onSearch?: (query: string) => void;
}

export const TopHeader: React.FC<TopHeaderProps> = ({ onSearch }) => {
  const [query, setQuery] = useState('');

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setQuery(e.target.value);
    if (onSearch) {
      onSearch(e.target.value);
    }
  };

  return (
    <header className="app-header">
      {/* Search Bar */}
      <div className="header-left">
        <div className="header-search">
          <Search size={14} className="header-search-icon" />
          <input
            type="text"
            className="header-search-input"
            placeholder="Search analysis, file name, or case ID..."
            value={query}
            onChange={handleSearchChange}
          />
        </div>
      </div>

      {/* Profile & Notifications */}
      <div className="header-right">
        <button className="header-btn-icon" title="Notifications" aria-label="Notifications">
          <Bell size={16} />
          <span className="notification-badge" />
        </button>

        <div className="user-profile-menu" title="User Profile">
          <div className="user-avatar">JD</div>
          <span className="user-name">John Doe</span>
          <ChevronDown size={14} className="user-chevron" />
        </div>
      </div>
    </header>
  );
};
