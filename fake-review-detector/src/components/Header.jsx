import { Link, useLocation } from 'react-router-dom';
import { FaSearch, FaFileUpload, FaChartBar, FaMoon, FaSun } from 'react-icons/fa';
import { useTheme } from '../context/ThemeContext';

export default function Header() {
  const location = useLocation();
  const { darkMode, toggleTheme } = useTheme();
  
  const navItems = [
    { path: "/", icon: <FaSearch />, label: "Single Analysis" },
    { path: "/batch", icon: <FaFileUpload />, label: "Batch Analysis" },
    { path: "/analytics", icon: <FaChartBar />, label: "Analytics" }
  ];

  return (
    <header className="bg-white dark:bg-gray-800 shadow-sm sticky top-0 z-10 transition-colors duration-300">
      <div className="container mx-auto px-4 py-4 flex justify-between items-center">
        <Link to="/" className="flex items-center">
          <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600 dark:from-indigo-400 dark:to-purple-400">
            Fake Review Detector
          </h1>
        </Link>
        <div className="flex items-center space-x-4">
          <nav className="flex space-x-2">
            {navItems.map((item) => (
              <Link 
                key={item.path}
                to={item.path} 
                className={`flex items-center px-4 py-2 rounded-lg transition ${
                  location.pathname === item.path
                    ? 'bg-indigo-100 text-indigo-700 dark:bg-gray-700 dark:text-indigo-300'
                    : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700'
                }`}
              >
                <span className="mr-2">{item.icon}</span>
                <span className="hidden sm:inline">{item.label}</span>
              </Link>
            ))}
          </nav>
          <button
            onClick={toggleTheme}
            className="p-2 rounded-full text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition"
            aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {darkMode ? <FaSun className="text-yellow-300" /> : <FaMoon />}
          </button>
        </div>
      </div>
    </header>
  );
}