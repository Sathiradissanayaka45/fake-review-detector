import StatsChart from './StatsChart';
import { motion } from 'framer-motion';

export default function ResultsDisplay({ result }) {
  if (!result) return null;

  // Adjust the prediction display based on your model's output
  const displayPrediction = result.prediction === 'fake' ? 'Fake Review' : 'Genuine Review';
  const predictionColorClass = result.prediction === 'fake' ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400';
  const confidenceColorClass = result.prediction === 'fake' ? 'text-red-600' : 'text-green-600';
  const confidenceBgClass = result.prediction === 'fake' ? 'bg-red-200' : 'bg-green-200';
  const confidenceBarClass = result.prediction === 'fake' ? 'bg-red-500' : 'bg-green-500';

  // Adjust warning flags display
  const warningFlags = result.explanation?.warning_flags || [];

  return (
    <div className="space-y-6">
      {/* Prediction section */}
      <div className="bg-white dark:bg-gray-700 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-gray-600">
        <h2 className="text-xl font-semibold mb-4 text-gray-800 dark:text-gray-200">Analysis Results</h2>
        
        <div className="flex flex-col sm:flex-row items-center mb-6 space-y-4 sm:space-y-0 sm:space-x-6">
          <div className={`text-2xl font-bold ${predictionColorClass}`}>
            {displayPrediction}
          </div>
          
          <div className="w-full sm:w-64">
            <div className="relative pt-1">
              <div className="flex items-center justify-between">
                <div>
                  <span className={`text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full ${confidenceBgClass} ${confidenceColorClass}`}>
                    Confidence
                  </span>
                </div>
                <div className="text-right">
                  <span className={`text-xs font-semibold inline-block ${confidenceColorClass}`}>
                    {Math.round(result.probability * 100)}%
                  </span>
                </div>
              </div>
              <div className={`overflow-hidden h-2 mb-4 text-xs flex rounded ${confidenceBgClass}`}>
                <div
                  style={{ width: `${result.probability * 100}%` }}
                  className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center ${confidenceBarClass}`}
                ></div>
              </div>
            </div>
          </div>
        </div>
        
        <StatsChart features={result.features} />
      </div>
      
      {/* Warning flags section */}

{warningFlags.length > 0 && (
  <div className="bg-white dark:bg-gray-700 rounded-xl shadow-lg p-6 border border-red-200 dark:border-red-400/30">
    <h3 className="text-lg font-medium text-gray-900 dark:text-gray-200 mb-3">Warning Flags</h3>
    <ul className="space-y-3">
      {warningFlags.map((flag, index) => (
        <li key={index} className="flex items-start">
          <span className="flex-shrink-0 h-5 w-5 text-red-500 mt-0.5">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          </span>
          <span className="ml-2 text-gray-700 dark:text-gray-300">
            {flag.description || flag.type || flag.toString()}
          </span>
        </li>
      ))}
    </ul>
  </div>
)}
      
      {/* Key features section */}
      <div className="bg-white dark:bg-gray-700 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-gray-600">
        <h3 className="text-lg font-medium text-gray-900 dark:text-gray-200 mb-4">Key Features</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-gray-50 dark:bg-gray-600 p-4 rounded-lg shadow-sm hover:shadow-md transition">
            <div className="text-sm text-gray-500 dark:text-gray-300">Word Count</div>
            <div className="text-2xl font-bold text-gray-800 dark:text-white">{result.features.word_count}</div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-600 p-4 rounded-lg shadow-sm hover:shadow-md transition">
            <div className="text-sm text-gray-500 dark:text-gray-300">Readability Score</div>
            <div className="text-2xl font-bold text-gray-800 dark:text-white">{Math.round(result.features.readability_score)}</div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-600 p-4 rounded-lg shadow-sm hover:shadow-md transition">
            <div className="text-sm text-gray-500 dark:text-gray-300">Sentiment Score</div>
            <div className="text-2xl font-bold text-gray-800 dark:text-white">{result.features.sentiment_score?.toFixed(2) || 'N/A'}</div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-600 p-4 rounded-lg shadow-sm hover:shadow-md transition">
            <div className="text-sm text-gray-500 dark:text-gray-300">First Person Pronouns</div>
            <div className="text-2xl font-bold text-gray-800 dark:text-white">{result.features.first_person_pronouns}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
