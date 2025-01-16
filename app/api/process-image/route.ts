import { NextRequest, NextResponse } from 'next/server'
import { verify } from 'jsonwebtoken'
import { getUserById } from '@/lib/auth'
import { exec } from 'child_process'
import { promisify } from 'util'
import fs from 'fs/promises'
import path from 'path'
import nodemailer from 'nodemailer'

const execAsync = promisify(exec)

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key'

export async function POST(req: NextRequest) {
  const authHeader = req.headers.get('authorization')
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const token = authHeader.split(' ')[1]
  try {
    const decoded = verify(token, JWT_SECRET) as { userId: string }
    const user = await getUserById(decoded.userId)
    if (!user) {
      return NextResponse.json({ error: 'User not found' }, { status: 401 })
    }
  } catch (error) {
    return NextResponse.json({ error: 'Invalid token' }, { status: 401 })
  }

  try {
    const formData = await req.formData()
    const file = formData.get('file') as File
    if (!file) {
      return NextResponse.json({ error: 'No file uploaded' }, { status: 400 })
    }

    const bytes = await file.arrayBuffer()
    const buffer = Buffer.from(bytes)

    // Save the uploaded file
    const uploadDir = path.join(process.cwd(), 'matlab')
    await fs.mkdir(uploadDir, { recursive: true })
    const filePath = path.join(uploadDir, 'input_image.png')
    await fs.writeFile(filePath, buffer)

    // Run MATLAB script
    const matlabPath = path.join(process.cwd(), 'matlab')
    const { stdout, stderr } = await execAsync(`matlab -nodisplay -nosplash -nodesktop -r "cd('${matlabPath}'); run('main.m');exit;"`)
    
    if (stderr) {
      console.error('MATLAB Error:', stderr)
      return NextResponse.json({ error: 'Error processing image' }, { status: 500 })
    }

    // Parse MATLAB output
    const results = JSON.parse(stdout)

    // Send email with results
    await sendEmail(results)

    // Send data to ThingSpeak
    await sendToThingSpeak(results)

    return NextResponse.json({ results })
  } catch (error) {
    console.error('Error:', error)
    return NextResponse.json({ error: 'Error processing image' }, { status: 500 })
  }
}

async function sendEmail(results: any) {
  const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
      user: process.env.EMAIL_USER,
      pass: process.env.EMAIL_PASS,
    },
  })

  const mailOptions = {
    from: process.env.EMAIL_USER,
    to: process.env.RECIPIENT_EMAIL,
    subject: 'Eye Disease Detection Results',
    text: JSON.stringify(results, null, 2),
    attachments: [
      { filename: 'DiagnosisReport_Diabetic_Retinopathy.pdf', path: './matlab/DiagnosisReport_Diabetic_Retinopathy.pdf' },
      { filename: 'DiagnosisReport_Exudates.pdf', path: './matlab/DiagnosisReport_Exudates.pdf' },
      { filename: 'DiagnosisReport_GLAUCOMA.pdf', path: './matlab/DiagnosisReport_GLAUCOMA.pdf' },
      { filename: 'DiagnosisReport_Macular_Edema.pdf', path: './matlab/DiagnosisReport_Macular_Edema.pdf' },
    ],
  }

  await transporter.sendMail(mailOptions)
}

async function sendToThingSpeak(results: any) {
  const Channel_ID = process.env.THINGSPEAK_CHANNEL_ID
  const Write_API_Key = process.env.THINGSPEAK_API_KEY

  const data = [
    results.dr.detected ? 1 : 0,
    getSeverityValue(results.dr.severity),
    results.me.detected ? 1 : 0,
    getSeverityValue(results.me.severity),
    results.glaucoma.detected ? 1 : 0,
    getSeverityValue(results.glaucoma.severity),
    results.exudates.detected ? 1 : 0,
    getSeverityValue(results.exudates.severity),
  ]

  const url = `https://api.thingspeak.com/update?api_key=${Write_API_Key}&field1=${data[0]}&field2=${data[1]}&field3=${data[2]}&field4=${data[3]}&field5=${data[4]}&field6=${data[5]}&field7=${data[6]}&field8=${data[7]}`

  await fetch(url)
}

function getSeverityValue(severity: string): number {
  switch (severity) {
    case 'Mild':
      return 0
    case 'Moderate':
      return 1
    case 'Proliferative':
      return 2
    case 'Severe':
      return 3
    default:
      return NaN
  }
}

