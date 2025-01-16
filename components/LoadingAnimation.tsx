import React from 'react'
import { Loader2 } from 'lucide-react'

const LoadingAnimation: React.FC = () => {
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
      <div className="bg-white p-6 rounded-lg shadow-xl flex flex-col items-center">
        <Loader2 className="w-16 h-16 text-blue-500 animate-spin mb-4" />
        <p className="text-lg font-semibold text-gray-700">Processing Image</p>
        <p className="text-sm text-gray-500 mt-2">Please wait while we analyze your retinal image...</p>
      </div>
    </div>
  )
}

export default LoadingAnimation

