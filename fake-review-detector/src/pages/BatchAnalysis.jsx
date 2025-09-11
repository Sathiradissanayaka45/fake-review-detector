import { useState } from 'react';
import BatchUpload from '../components/BatchUpload';
import BatchResults from '../components/BatchResults';
import { motion } from 'framer-motion';

export default function BatchAnalysis() {
  const [results, setResults] = useState(null);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="max-w-6xl mx-auto"
    >
      <div className="text-center mb-10">
        <h1 className="text-4xl font-bold text-gray-900 mb-3 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-cyan-600">
          Batch Review Analysis
        </h1>
        <p className="text-gray-600 max-w-lg mx-auto">
          Upload a CSV file containing multiple reviews to analyze them in bulk.
        </p>
      </div>
      
      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.2 }}
      >
        <BatchUpload onResults={setResults} />
      </motion.div>
      
      {results && (
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="mt-8"
        >
          <BatchResults results={results} />
        </motion.div>
      )}
    </motion.div>
  );
}