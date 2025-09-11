import { Doughnut } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';

ChartJS.register(ArcElement, Tooltip, Legend);

export default function StatsChart({ features }) {
  if (!features) return null;

  const data = {
    labels: ['Adjectives', 'Adverbs', 'Nouns', 'Verbs'],
    datasets: [
      {
        data: [
          features.adj_ratio * 100,
          features.adv_ratio * 100,
          features.noun_ratio * 100,
          features.verb_ratio * 100
        ],
        backgroundColor: [
          'rgba(255, 99, 132, 0.7)',
          'rgba(54, 162, 235, 0.7)',
          'rgba(255, 206, 86, 0.7)',
          'rgba(75, 192, 192, 0.7)'
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(75, 192, 192, 1)'
        ],
        borderWidth: 2,
      },
    ],
  };

  const options = {
    responsive: true,
    cutout: '70%',
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          color: '#6B7280',
          font: {
            family: 'Inter, sans-serif',
            size: 12
          }
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `${context.label}: ${context.raw.toFixed(1)}%`;
          }
        }
      }
    },
  };

  return (
    <div className="max-w-xs mx-auto">
      <div className="relative">
        <Doughnut data={data} options={options} />
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <span className="text-2xl font-bold text-gray-800 dark:text-gray-200">
              {Math.round(
                (features.adj_ratio + features.adv_ratio + 
                 features.noun_ratio + features.verb_ratio) * 100
              )}%
            </span>
            <p className="text-xs text-gray-500 dark:text-gray-400">POS Coverage</p>
          </div>
        </div>
      </div>
      <h3 className="text-lg font-medium text-gray-900 dark:text-gray-200 mb-2 text-center mt-4">
        Parts of Speech
      </h3>
    </div>
  );
}