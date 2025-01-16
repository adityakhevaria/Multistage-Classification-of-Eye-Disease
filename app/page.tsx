'use client'

import { useState, useEffect } from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Upload, Eye, Loader } from 'lucide-react'
import ResultCard from '@/components/ResultCard'
import LoginForm from '@/components/LoginForm'
import SignupForm from '@/components/SignupForm'
import LoadingAnimation from '@/components/LoadingAnimation'

export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<any | null>(null)
  const [token, setToken] = useState<string | null>(null)
  const [showLogin, setShowLogin] = useState(true)

  useEffect(() => {
    const storedToken = localStorage.getItem('token')
    if (storedToken) {
      setToken(storedToken)
    }
  }, [])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0])
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file || !token) return

    setLoading(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('/api/process-image', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData,
      })
      const data = await response.json()
      setResults(data.results)
    } catch (error) {
      console.error('Error:', error)
      setResults(null)
    } finally {
      setLoading(false)
    }
  }

  const handleLogin = (newToken: string) => {
    setToken(newToken)
    localStorage.setItem('token', newToken)
  }

  const handleLogout = () => {
    setToken(null)
    localStorage.removeItem('token')
  }

  if (!token) {
    return (
      <main className="min-h-screen bg-gradient-to-b from-blue-100 to-white flex items-center justify-center">
        <div className="w-full max-w-md">
          {showLogin ? (
            <>
              <LoginForm onLogin={handleLogin} />
              <p className="text-center mt-4">
                Don't have an account?{' '}
                <Button variant="link" onClick={() => setShowLogin(false)}>Sign up</Button>
              </p>
            </>
          ) : (
            <>
              <SignupForm onSignup={() => setShowLogin(true)} />
              <p className="text-center mt-4">
                Already have an account?{' '}
                <Button variant="link" onClick={() => setShowLogin(true)}>Login</Button>
              </p>
            </>
          )}
        </div>
      </main>
    )
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-blue-100 to-white">
      <div className="container mx-auto px-4 py-8">
        <header className="flex justify-between items-center mb-12">
          <div>
            <h1 className="text-4xl font-bold text-blue-800 mb-2">Eye Diagnosis AI</h1>
            <p className="text-xl text-gray-600">Advanced retinal analysis for early disease detection</p>
          </div>
          <Button onClick={handleLogout}>Logout</Button>
        </header>

        <Card className="w-full max-w-3xl mx-auto mb-8">
          <CardHeader>
            <CardTitle className="text-2xl text-center">Upload Retinal Image</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="flex items-center justify-center w-full">
                <label htmlFor="dropzone-file" className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100">
                  <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    <Upload className="w-10 h-10 mb-3 text-gray-400" />
                    <p className="mb-2 text-sm text-gray-500"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                    <p className="text-xs text-gray-500">PNG, JPG or TIFF (MAX. 800x400px)</p>
                  </div>
                  <Input id="dropzone-file" type="file" className="hidden" onChange={handleFileChange} accept="image/*" />
                </label>
              </div>
              {file && <p className="text-sm text-gray-500">Selected file: {file.name}</p>}
              <Button type="submit" disabled={!file || loading} className="w-full">
                {loading ? 'Processing...' : 'Analyze Image'}
              </Button>
            </form>
          </CardContent>
        </Card>

        {loading && <LoadingAnimation />}

        {results && (
          <Card className="w-full max-w-4xl mx-auto">
            <CardHeader>
              <CardTitle className="text-2xl text-center">Analysis Results</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="dr" className="w-full">
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="dr">Diabetic Retinopathy</TabsTrigger>
                  <TabsTrigger value="me">Macular Edema</TabsTrigger>
                  <TabsTrigger value="glaucoma">Glaucoma</TabsTrigger>
                  <TabsTrigger value="exudates">Exudates</TabsTrigger>
                </TabsList>
                <TabsContent value="dr">
                  <ResultCard 
                    title="Diabetic Retinopathy" 
                    data={results.dr}
                    description="Diabetic retinopathy is a diabetes complication that affects eyes. It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina)."
                  />
                </TabsContent>
                <TabsContent value="me">
                  <ResultCard 
                    title="Macular Edema" 
                    data={results.me}
                    description="Macular edema occurs when fluid builds up in the macula, causing swelling. The macula is the part of the retina responsible for sharp, straight-ahead vision."
                  />
                </TabsContent>
                <TabsContent value="glaucoma">
                  <ResultCard 
                    title="Glaucoma" 
                    data={results.glaucoma}
                    description="Glaucoma is a group of eye conditions that damage the optic nerve, which is vital for good vision. This damage is often caused by an abnormally high pressure in your eye."
                  />
                </TabsContent>
                <TabsContent valueHere's the continuation of the text stream:

<cut_off_point>
eye."
                  />
                </TabsContent>
                <TabsContent value
</cut_off_point>

="exudates">
                  <ResultCard 
                    title="Exudates" 
                    data={results.exudates}
                    description="Exudates are yellowish deposits of lipid and protein in the retina. They are often associated with diabetic retinopathy and can indicate the severity of the condition."
                  />
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        )}
      </div>
    </main>
  )
}

