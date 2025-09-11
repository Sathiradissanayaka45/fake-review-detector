import { useState } from 'react';
import axios from 'axios';
import { FaFileCsv, FaSpinner } from 'react-icons/fa';
import { toast } from 'react-hot-toast';

export default function BatchUpload({ onResults }) {
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file');
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/batch_predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      if (response.data.error) {
        throw new Error(response.data.message || 'Batch analysis failed');
      }

      // Transform each result to match your frontend expectations
      const transformedResults = response.data.results.map(result => ({
        prediction: result.prediction,
        probability: result.probability,
        features: {
          text: result.features?.text || '',
          word_count: result.features.word_count,
          readability_score: result.features.readability_score,
          sentiment_score: result.features.sentiment_score,
          first_person_pronouns: result.features.first_person_pronouns,
          // Add other features as needed
        }
      }));

      onResults(transformedResults);
      toast.success(`Analyzed ${transformedResults.length} reviews`);
    } catch (err) {
      setError(err.response?.data?.message || err.message || 'An error occurred');
      toast.error('Failed to analyze batch');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-semibold mb-4">Batch Analyze Reviews</h2>
      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Upload CSV File
          </label>
          <div className="flex items-center">
            <label className="flex flex-col items-center px-4 py-6 bg-white rounded-md border border-dashed border-gray-300 cursor-pointer hover:bg-gray-50">
              <FaFileCsv className="text-indigo-600 text-2xl mb-2" />
              <span className="text-sm text-gray-600">
                {file ? file.name : 'Choose a CSV file'}
              </span>
              <input 
                type="file" 
                className="hidden" 
                accept=".csv"
                onChange={handleFileChange}
              />
            </label>
          </div>
          <p className="mt-1 text-xs text-gray-500">
            CSV should contain at least a "text" column, and optionally a "rating" column
          </p>
        </div>
        
        <button
          type="submit"
          disabled={isLoading || !file}
          className="w-full bg-indigo-600 text-white py-2 px-4 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <span className="flex items-center justify-center">
              <FaSpinner className="animate-spin mr-2" /> Analyzing...
            </span>
          ) : 'Analyze Batch'}
        </button>
        
        {error && (
          <div className="mt-4 text-red-600 text-sm">
            {error}
          </div>
        )}
      </form>
    </div>
  );
}