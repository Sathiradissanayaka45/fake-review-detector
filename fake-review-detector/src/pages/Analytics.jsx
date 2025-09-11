import { useEffect, useState } from 'react';
import axios from 'axios';
import { Bar, Pie, Line } from 'react-chartjs-2';
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  BarElement, 
  Title, 
  Tooltip, 
  Legend,
  ArcElement,
  PointElement,
  LineElement
} from 'chart.js';
import { motion } from 'framer-motion';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement
);

export default function Analytics() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const response = await axios.get('http://localhost:5000/analytics');
        if (response.data.success) {
          setStats(response.data.analytics);
        } else {
          setError(response.data.message || 'Failed to load analytics');
        }
      } catch (error) {
        console.error('Error fetching analytics:', error);
        setError(error.response?.data?.message || 'Failed to connect to server');
      } finally {
        setLoading(false);
      }
    };
    
    fetchAnalytics();
  }, []);

  if (loading) return (
    <div className="flex items-center justify-center h-64">
      <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-600"></div>
    </div>
  );

  if (error) return (
    <div className="text-center py-12">
      <div className="text-red-500 mb-4">{error}</div>
      <button 
        onClick={() => window.location.reload()}
        className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
      >
        Retry
      </button>
    </div>
  );

  if (!stats) return (
    <div className="text-center py-12">
      <div className="text-gray-500 mb-4">No analytics data available</div>
    </div>
  );

  // Prepare data for charts
  const predictionData = {
    labels: stats.prediction_stats.map(item => item.prediction === 'fake' ? 'Fake' : 'Genuine'),
    datasets: [{
      data: stats.prediction_stats.map(item => item.count),
      backgroundColor: [
        'rgba(239, 68, 68, 0.7)',
        'rgba(16, 185, 129, 0.7)'
      ],
      borderColor: [
        'rgba(239, 68, 68, 1)',
        'rgba(16, 185, 129, 1)'
      ],
      borderWidth: 2
    }]
  };

  const sentimentData = {
    labels: stats.sentiment_stats.map(item => item.sentiment),
    datasets: [{
      label: 'Reviews',
      data: stats.sentiment_stats.map(item => item.count),
      backgroundColor: [
        'rgba(239, 68, 68, 0.7)',
        'rgba(156, 163, 175, 0.7)',
        'rgba(16, 185, 129, 0.7)'
      ],
      borderColor: [
        'rgba(239, 68, 68, 1)',
        'rgba(156, 163, 175, 1)',
        'rgba(16, 185, 129, 1)'
      ],
      borderWidth: 2
    }]
  };

  // Group daily trend by date
  const dailyTrendData = {
    labels: [...new Set(stats.daily_trend.map(item => item.date))],
    datasets: [
      {
        label: 'Fake Reviews',
        data: stats.daily_trend
          .filter(item => item.prediction === 'fake')
          .map(item => item.count),
        borderColor: 'rgba(239, 68, 68, 1)',
        backgroundColor: 'rgba(239, 68, 68, 0.2)',
        tension: 0.3,
        borderWidth: 2
      },
      {
        label: 'Genuine Reviews',
        data: stats.daily_trend
          .filter(item => item.prediction === 'genuine')
          .map(item => item.count),
        borderColor: 'rgba(16, 185, 129, 1)',
        backgroundColor: 'rgba(16, 185, 129, 0.2)',
        tension: 0.3,
        borderWidth: 2
      }
    ]
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="container mx-auto px-4 py-8"
    >
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-3 bg-clip-text text-transparent bg-gradient-to-r from-purple-600 to-pink-600">
          Review Analytics Dashboard
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Comprehensive insights and statistics about analyzed reviews
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        {/* Prediction Distribution */}
        <motion.div 
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-white p-6 rounded-xl shadow-lg border border-gray-100"
        >
          <h2 className="text-xl font-semibold mb-4 text-gray-800">Review Authenticity</h2>
          <div className="h-64">
            <Pie
              data={predictionData}
              options={{
                responsive: true,
                plugins: {
                  legend: { 
                    position: 'bottom'
                  },
                  tooltip: {
                    callbacks: {
                      label: (context) => {
                        const item = stats.prediction_stats[context.dataIndex];
                        return `${context.label}: ${item.count} (${item.percentage}%)`;
                      }
                    }
                  }
                }
              }}
            />
          </div>
        </motion.div>

        {/* Sentiment Analysis */}
        <motion.div 
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="bg-white p-6 rounded-xl shadow-lg border border-gray-100"
        >
          <h2 className="text-xl font-semibold mb-4 text-gray-800">Sentiment Distribution</h2>
          <div className="h-64">
            <Bar
              data={sentimentData}
              options={{
                responsive: true,
                scales: {
                  y: { beginAtZero: true }
                },
                plugins: {
                  legend: { display: false }
                }
              }}
            />
          </div>
        </motion.div>
      </div>

      {/* Daily Trend */}
      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.4 }}
        className="bg-white p-6 rounded-xl shadow-lg border border-gray-100 mb-8"
      >
        <h2 className="text-xl font-semibold mb-4 text-gray-800">Daily Review Trend</h2>
        <div className="h-96">
          <Line
            data={dailyTrendData}
            options={{
              responsive: true,
              scales: {
                y: { beginAtZero: true }
              },
              plugins: {
                legend: {
                  position: 'bottom'
                }
              }
            }}
          />
        </div>
      </motion.div>

      {/* Recent Reviews Table */}
      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="bg-white p-6 rounded-xl shadow-lg border border-gray-100"
      >
        <h2 className="text-xl font-semibold mb-4 text-gray-800">Recent Reviews</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Review</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Rating</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Prediction</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {stats.recent_reviews.map((review, index) => (
                <tr key={index} className="hover:bg-gray-50 transition">
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 max-w-xs truncate">
                    {review.review_text}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {review.rating}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                      review.prediction === 'fake' 
                        ? 'bg-red-100 text-red-800' 
                        : 'bg-green-100 text-green-800'
                    }`}>
                      {review.prediction === 'fake' ? 'Fake' : 'Genuine'}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {Math.round(review.probability * 100)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>
    </motion.div>
  );
}