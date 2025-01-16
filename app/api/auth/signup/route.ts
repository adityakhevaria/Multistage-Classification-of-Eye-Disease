import { NextRequest, NextResponse } from 'next/server'
import { createUser } from '@/lib/auth'

export async function POST(req: NextRequest) {
  try {
    const { email, password } = await req.json()

    if (!email || !password) {
      return NextResponse.json({ error: 'Missing email or password' }, { status: 400 })
    }

    const user = await createUser(email, password)
    return NextResponse.json({ message: 'User created successfully', userId: user._id })
  } catch (error) {
    console.error('Signup error:', error)
    return NextResponse.json({ error: error.message || 'Error creating user' }, { status: 500 })
  }
}

