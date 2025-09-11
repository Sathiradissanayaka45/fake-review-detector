import { useState } from 'react';
import axios from 'axios';
import { FaSpinner } from 'react-icons/fa';
import { toast } from 'react-hot-toast';

export default function ReviewForm({ onAnalysisComplete }) {
  const [text, setText] = useState('');
  const [rating, setRating] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isFocused, setIsFocused] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) {
      toast.error('Please enter review text');
      return;
    }
    
    setIsLoading(true);
    
    try {
      const response = await axios.post('http://localhost:5000/predict', {
        text,
        rating: rating ? parseInt(rating) : null
      });

      if (response.data.error) {
        throw new Error(response.data.message || 'Analysis failed');
      }

      // Transform the response to match your frontend expectations
      const transformedResult = {
        prediction: response.data.result.prediction,
        probability: response.data.result.probability,
        features: response.data.result.features,
        explanation: {
          key_factors: {
            readability_score: response.data.result.features.readability_score,
            sentiment_score: response.data.result.features.sentiment_score,
            word_count: response.data.result.features.word_count,
            first_person_pronouns: response.data.result.features.first_person_pronouns
          },
          warning_flags: response.data.result.explanation.warning_flags || []
        }
      };

      onAnalysisComplete(transformedResult);
      toast.success('Analysis completed!');
    } catch (err) {
      toast.error(err.response?.data?.message || err.message || 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <h2 className="text-xl font-semibold mb-6 text-gray-800 dark:text-gray-200">Analyze a Review</h2>
      <form onSubmit={handleSubmit}>
        <div className="relative mb-6">
          <textarea
            id="review-text"
            rows="5"
            className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-200"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            required
          />
          <label 
            htmlFor="review-text"
            className={`absolute left-3 transition-all duration-200 pointer-events-none ${
              isFocused || text ? 
                '-top-3 text-xs bg-white dark:bg-gray-700 px-1 text-indigo-600 dark:text-indigo-400' : 
                'top-3 text-gray-500 dark:text-gray-400'
            }`}
          >
            Review Text
          </label>
        </div>
        
        <div className="mb-8">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Rating (Optional)
          </label>
          <div className="flex items-center justify-center space-x-2">
            {[1, 2, 3, 4, 5].map((num) => (
              <button
                key={num}
                type="button"
                onClick={() => setRating(num.toString())}
                className={`w-12 h-12 flex items-center justify-center rounded-full transition-all duration-200 transform hover:scale-110 ${
                  rating === num.toString()
                    ? 'bg-indigo-600 text-white shadow-lg'
                    : 'bg-gray-100 dark:bg-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-500'
                }`}
              >
                {num}
              </button>
            ))}
          </div>
        </div>
        
        <button
          type="submit"
          disabled={isLoading}
          className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white py-3 px-4 rounded-lg hover:from-indigo-700 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-70 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-[1.01] flex items-center justify-center shadow-lg"
        >
          {isLoading ? (
            <>
              <FaSpinner className="animate-spin mr-2" />
              Analyzing...
            </>
          ) : (
            <span className="flex items-center">
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
              </svg>
              Analyze Review
            </span>
          )}
        </button>
      </form>
    </div>
  );
}