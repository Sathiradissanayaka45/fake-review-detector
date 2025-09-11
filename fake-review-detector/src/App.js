// App.js
import { Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import BatchAnalysis from './pages/BatchAnalysis';
import Analytics from './pages/Analytics';
import Header from './components/Header';
import { Toaster } from 'react-hot-toast';
import { ThemeProvider } from './context/ThemeContext';
import Footer from './components/Footer';

function App() {
  return (
    <ThemeProvider>
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 transition-colors duration-300">
        <Toaster position="top-right" />
        <Header />
        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/batch" element={<BatchAnalysis />} />
            <Route path="/analytics" element={<Analytics />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </ThemeProvider>
  );
}

export default App;