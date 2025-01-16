import { NextRequest, NextResponse } from 'next/server'
import { getUser } from '@/lib/auth'
import { sign } from 'jsonwebtoken'

const JWT_SECRET = process.env.JWT_SECRET
if (!JWT_SECRET) {
  throw new Error('JWT_SECRET is not set in the environment variables. Please set this in your .env.local file.')
}

export async function POST(req: NextRequest) {
  try {
    const { email, password } = await req.json()

    if (!email || !password) {
      return NextResponse.json({ error: 'Missing email or password' }, { status: 400 })
    }

    const user = await getUser(email, password)
    if (user) {
      const token = sign({ userId: user._id }, JWT_SECRET, { expiresIn: '1h' })
      return NextResponse.json({ token })
    } else {
      return NextResponse.json({ error: 'Invalid credentials' }, { status: 401 })
    }
  } catch (error) {
    console.error('Login error:', error)
    return NextResponse.json({ error: 'Error during login' }, { status: 500 })
  }
}

