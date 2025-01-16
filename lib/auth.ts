import { hash, compare } from 'bcryptjs'
import { ObjectId } from 'mongodb'
import clientPromise from './db'

export interface User {
  _id: ObjectId
  email: string
  password: string
}

export async function createUser(email: string, password: string): Promise<User> {
  const client = await clientPromise
  const db = client.db()

  const existingUser = await db.collection('users').findOne({ email })
  if (existingUser) {
    throw new Error('User already exists')
  }

  const hashedPassword = await hash(password, 12)
  const result = await db.collection('users').insertOne({ email, password: hashedPassword })
  return { _id: result.insertedId, email, password: hashedPassword }
}

export async function getUser(email: string, password: string): Promise<User | null> {
  const client = await clientPromise
  const db = client.db()

  const user = await db.collection('users').findOne({ email })
  if (user && await compare(password, user.password)) {
    return user as User
  }
  return null
}

export async function getUserById(id: string): Promise<User | null> {
  const client = await clientPromise
  const db = client.db()

  const user = await db.collection('users').findOne({ _id: new ObjectId(id) })
  return user as User | null
}

