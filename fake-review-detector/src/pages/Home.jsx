import { useState } from 'react';
import ReviewForm from '../components/ReviewForm';
import ResultsDisplay from '../components/ResultsDisplay';
import { motion } from 'framer-motion';

export default function Home() {
  const [result, setResult] = useState(null);

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="max-w-4xl mx-auto"
    >
      <div className="text-center mb-10">
        <motion.h1 
          initial={{ y: -20 }}
          animate={{ y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-4xl font-bold text-gray-900 dark:text-white mb-3 bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600"
        >
          Single Review Analysis
        </motion.h1>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-gray-600 dark:text-gray-300 max-w-lg mx-auto"
        >
          Analyze individual product reviews to detect authenticity with our advanced AI model.
        </motion.p>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <motion.div 
          initial={{ x: -20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="lg:col-span-2"
        >
          <div className="bg-white dark:bg-gray-800 backdrop-blur-lg bg-opacity-80 dark:bg-opacity-80 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6 transition-all duration-300 hover:shadow-xl">
            <ReviewForm onAnalysisComplete={setResult} />
          </div>
        </motion.div>
        
        {result && (
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="lg:col-span-2"
          >
            <div className="bg-white dark:bg-gray-800 backdrop-blur-lg bg-opacity-80 dark:bg-opacity-80 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6 transition-all duration-300 hover:shadow-xl">
              <ResultsDisplay result={result} />
            </div>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}