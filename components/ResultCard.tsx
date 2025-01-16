import Image from 'next/image'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { AlertTriangle, CheckCircle } from 'lucide-react'

interface ResultCardProps {
  title: string
  data: {
    detected: boolean
    severity: string
    metrics: {
      Accuracy: number
      PSNR: number
      Entropy: number
      AUC: number
      Precision: number
      Recall: number
      F1_Score: number
      Specificity: number
      Sensitivity: number
    }
  }
  description: string
}

export default function ResultCard({ title, data, description }: ResultCardProps) {
  return (
    <Card className="overflow-hidden">
      <CardHeader className="bg-gradient-to-r from-blue-500 to-purple-600 text-white">
        <div className="flex items-center justify-between">
          <CardTitle className="text-2xl font-bold">{title}</CardTitle>
          {data.detected ? (
            <AlertTriangle className="w-8 h-8 text-yellow-300" />
          ) : (
            <CheckCircle className="w-8 h-8 text-green-300" />
          )}
        </div>
        <CardDescription className="text-gray-200 mt-2">{description}</CardDescription>
      </CardHeader>
      <CardContent className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div>
            <h3 className="font-semibold text-xl mb-4">Diagnosis:</h3>
            <p className={`text-lg font-medium ${data.detected ? 'text-red-500' : 'text-green-500'}`}>
              {data.detected ? `Detected (${data.severity})` : 'Not Detected'}
            </p>
          </div>
          <div>
            <h3 className="font-semibold text-xl mb-4">Metrics:</h3>
            <div className="grid grid-cols-2 gap-4">
              {Object.entries(data.metrics).map(([key, value]) => (
                <div key={key} className="bg-gray-100 p-3 rounded-md">
                  <p className="font-medium text-gray-700">{key}</p>
                  <p className="text-lg font-semibold text-blue-600">
                    {typeof value === 'number' ? value.toFixed(2) : value}
                    {key.includes('Accuracy') || key.includes('Precision') || key.includes('Recall') || key.includes('Specificity') || key.includes('Sensitivity') ? '%' : ''}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

