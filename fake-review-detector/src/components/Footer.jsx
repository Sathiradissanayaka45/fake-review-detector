export default function Footer() {
    return (
      <footer className="bg-white dark:bg-gray-800 py-6 border-t border-gray-200 dark:border-gray-700 mt-12 transition-colors duration-300">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="text-center md:text-left mb-4 md:mb-0">
              <p className="text-gray-500 dark:text-gray-400 text-sm">
                © {new Date().getFullYear()} Fake Review Detector - All rights reserved
              </p>
            </div>
            <div className="flex space-x-4">
              <a href="#" className="text-gray-500 dark:text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 transition">
                Privacy Policy
              </a>
              <a href="#" className="text-gray-500 dark:text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 transition">
                Terms of Service
              </a>
              <a href="#" className="text-gray-500 dark:text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 transition">
                Contact Us
              </a>
            </div>
          </div>
        </div>
      </footer>
    );
  }