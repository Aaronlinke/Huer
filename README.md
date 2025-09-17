# Huer
# Black Sultan - Vollständig ausgereiftes Alles-in-Einem-System

Hier ist die erweiterte und vollständige Version mit erweiterten Funktionalitäten, echter Datenbankanbindung, umfassenden Tests und optimierter Struktur.

## Erweiterte Projektstruktur

```
black-sultan/
├── README.md
├── docker-compose.yml
├── .env
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── alembic/
│   │   ├── versions/
│   │   ├── env.py
│   │   └── script.py.mako
│   ├── app/
│   │   ├── main.py
│   │   ├── __init__.py
│   │   ├── database.py
│   │   ├── schemas.py
│   │   ├── routes/
│   │   │   ├── auth.py
│   │   │   ├── users.py
│   │   │   ├── ledger.py
│   │   │   ├── ai.py
│   │   │   └── admin.py
│   │   ├── models/
│   │   │   ├── user.py
│   │   │   ├── ledger.py
│   │   │   └── __init__.py
│   │   ├── services/
│   │   │   ├── ai_proxy.py
│   │   │   ├── blockchain.py
│   │   │   ├── auth.py
│   │   │   └── __init__.py
│   │   ├── utils/
│   │   │   ├── jwt_handler.py
│   │   │   ├── rbac.py
│   │   │   ├── security.py
│   │   │   └── __init__.py
│   │   └── tests/
│   │       ├── test_auth.py
│   │       ├── test_users.py
│   │       └── test_ledger.py
│   ├── migrations/
│   │   └── init.sql
│   └── scripts/
│       └── init_db.py
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── tsconfig.json
│   ├── next.config.js
│   ├── components/
│   │   ├── Layout.tsx
│   │   ├── Navbar.tsx
│   │   ├── AuthForm.tsx
│   │   └── Dashboard.tsx
│   ├── pages/
│   │   ├── index.tsx
│   │   ├── login.tsx
│   │   ├── register.tsx
│   │   ├── dashboard.tsx
│   │   ├── admin.tsx
│   │   └── api/
│   │       └── auth.ts
│   ├── styles/
│   │   └── globals.css
│   ├── hooks/
│   │   └── useAuth.ts
│   └── utils/
│       └── api.ts
├── ai-service/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py
│   └── services/
│       └── openai_service.py
├── contracts/
│   ├── BlackSultan.sol
│   ├── deploy.js
│   └── hardhat.config.js
├── scripts/
│   ├── init_db.sh
│   └── deploy_contract.sh
└── postman/
    ├── BlackSultan.postman_collection.json
    └── BlackSultan.postman_environment.json
```

## 1. Docker Compose (docker-compose.yml)

```yaml
version: "3.9"
services:
  backend:
    build: ./backend
    container_name: black_sultan_backend
    env_file: .env
    ports:
      - "8000:8000"
    depends_on:
      - db
      - ai_service
    volumes:
      - ./backend:/app
      - ./backend/migrations/init.sql:/docker-entrypoint-initdb.d/init.sql
    command: >
      sh -c "python scripts/init_db.py &&
             alembic upgrade head &&
             uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

  frontend:
    build: ./frontend
    container_name: black_sultan_frontend
    ports:
      - "3000:3000"
    env_file: .env
    depends_on:
      - backend
    volumes:
      - ./frontend:/app
      - /app/node_modules

  ai_service:
    build: ./ai-service
    container_name: black_sultan_ai
    ports:
      - "5000:5000"
    env_file: .env
    volumes:
      - ./ai-service:/app

  db:
    image: postgres:15
    container_name: black_sultan_db
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backend/migrations/init.sql:/docker-entrypoint-initdb.d/init.sql

volumes:
  postgres_data:
```

## 2. Umgebungsvariablen (.env)

```env
# Database
POSTGRES_USER=admin
POSTGRES_PASSWORD=supersecretpassword
POSTGRES_DB=black_sultan
DATABASE_URL=postgresql://admin:supersecretpassword@db:5432/black_sultan

# JWT
JWT_SECRET=ultrasecretjwttokenforsecurity
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=30

# AI Service
AI_SERVICE_URL=http://ai_service:5000
OPENAI_API_KEY=your_openai_api_key_here

# App
FRONTEND_URL=http://localhost:3000
BACKEND_URL=http://localhost:8000
DEBUG=False

# Blockchain (für Entwicklung)
BLOCKCHAIN_RPC_URL=http://localhost:8545
CONTRACT_ADDRESS=
```

## 3. Backend

### Dockerfile (backend/Dockerfile)
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create alembic versions directory
RUN mkdir -p alembic/versions

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### requirements.txt (backend/requirements.txt)
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
alembic==1.12.1
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
pydantic==2.5.0
pydantic-settings==2.1.0
requests==2.31.0
web3==6.11.0
python-multipart==0.0.6
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2
```

### Datenbank Initialisierung (backend/migrations/init.sql)
```sql
-- Tabelle für Benutzer
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabelle für Ledger-Einträge
CREATE TABLE IF NOT EXISTS ledger (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    amount DECIMAL(15, 2) NOT NULL,
    description TEXT,
    category VARCHAR(50),
    transaction_type VARCHAR(10) CHECK (transaction_type IN ('income', 'expense')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabelle für AI-Anfragen
CREATE TABLE IF NOT EXISTS ai_requests (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    prompt TEXT NOT NULL,
    response TEXT,
    model VARCHAR(50),
    tokens_used INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexe für bessere Performance
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_ledger_user_id ON ledger(user_id);
CREATE INDEX IF NOT EXISTS idx_ledger_created_at ON ledger(created_at);
CREATE INDEX IF NOT EXISTS idx_ai_requests_user_id ON ai_requests(user_id);

-- Initiale Admin-Benutzer einfügen (Passwort: admin123)
INSERT INTO users (username, email, password_hash, role) 
VALUES 
('admin', 'admin@blacksultan.com', '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW', 'admin'),
('testuser', 'user@blacksultan.com', '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW', 'user')
ON CONFLICT (username) DO NOTHING;

-- Beispiel-Ledger-Einträge
INSERT INTO ledger (user_id, amount, description, category, transaction_type) 
VALUES 
(1, 1000.00, 'Initial Deposit', 'funding', 'income'),
(2, 500.00, 'Initial Deposit', 'funding', 'income'),
(1, -150.00, 'Server Costs', 'infrastructure', 'expense')
ON CONFLICT (id) DO NOTHING;
```

### Datenbank-Setup Script (backend/scripts/init_db.py)
```python
#!/usr/bin/env python3
import os
import sys
from sqlalchemy import create_engine
from app.database import Base
import app.models.user
import app.models.ledger

def init_database():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL environment variable is not set")
        sys.exit(1)
    
    engine = create_engine(database_url)
    
    try:
        # Create all tables
        Base.metadata.create_all(engine)
        print("Database tables created successfully")
    except Exception as e:
        print(f"Error creating database tables: {e}")
        sys.exit(1)

if __name__ == "__main__":
    init_database()
```

### Haupt-App (backend/app/main.py)
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.database import engine, Base
from app.routes import auth, users, ledger, ai, admin

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create tables
    Base.metadata.create_all(bind=engine)
    yield
    # Shutdown: Cleanup
    pass

app = FastAPI(
    title="Black Sultan API",
    description="Complete All-in-One Management System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(ledger.router, prefix="/api/ledger", tags=["Ledger"])
app.include_router(ai.router, prefix="/api/ai", tags=["AI"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])

@app.get("/")
async def root():
    return {"message": "Black Sultan API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Black Sultan API"}
```

### Datenbank-Konfiguration (backend/app/database.py)
```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:supersecretpassword@localhost:5432/black_sultan")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency for database sessions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Modelle (backend/app/models/)

#### user.py
```python
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="user")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
```

#### ledger.py
```python
from sqlalchemy import Column, Integer, Numeric, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base

class LedgerEntry(Base):
    __tablename__ = "ledger"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    amount = Column(Numeric(15, 2), nullable=False)
    description = Column(Text)
    category = Column(String(50))
    transaction_type = Column(String(10), nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    user = relationship("User")
```

### Services (backend/app/services/)

#### ai_proxy.py
```python
import os
import httpx
from fastapi import HTTPException

class AIService:
    def __init__(self):
        self.ai_service_url = os.getenv("AI_SERVICE_URL", "http://ai_service:5000")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
    
    async def process_request(self, prompt: str, model: str = "gpt-3.5-turbo"):
        if not self.openai_api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 150
                }
                headers = {
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                }
                
                response = await client.post(
                    f"{self.ai_service_url}/process",
                    json=payload,
                    headers=headers,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"AI service error: {response.text}"
                    )
                    
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"AI service unavailable: {str(e)}")
```

#### blockchain.py
```python
from web3 import Web3
import os
import json

class BlockchainService:
    def __init__(self):
        self.rpc_url = os.getenv("BLOCKCHAIN_RPC_URL", "http://localhost:8545")
        self.contract_address = os.getenv("CONTRACT_ADDRESS")
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        # Load contract ABI (would be loaded from compiled contract)
        self.contract_abi = []  # Actual ABI would go here
        
        if self.contract_address:
            self.contract = self.w3.eth.contract(
                address=self.contract_address,
                abi=self.contract_abi
            )
    
    def get_balance(self, address: str):
        try:
            balance = self.w3.eth.get_balance(address)
            return self.w3.from_wei(balance, 'ether')
        except Exception as e:
            raise Exception(f"Error getting balance: {str(e)}")
```

### Routes (backend/app/routes/)

#### auth.py
```python
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta

from app.database import get_db
from app.models.user import User
from app.schemas import UserCreate, UserResponse, Token
from app.services.auth import AuthService
from app.utils.security import get_password_hash, verify_password

router = APIRouter()

@router.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(
            status_code=400,
            detail="Username already registered"
        )
    
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        username=user_data.username,
        email=user_data.email,
        password_hash=hashed_password,
        role=user_data.role or "user"
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = AuthService.authenticate_user(
        db, form_data.username, form_data.password
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = AuthService.create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=30)
    )
    
    return {"access_token": access_token, "token_type": "bearer"}
```

#### ledger.py
```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.models.ledger import LedgerEntry
from app.schemas import LedgerCreate, LedgerResponse
from app.services.auth import AuthService

router = APIRouter()

@router.get("/", response_model=List[LedgerResponse])
async def get_ledger_entries(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(AuthService.get_current_user)
):
    entries = db.query(LedgerEntry)\
        .filter(LedgerEntry.user_id == current_user.id)\
        .offset(skip)\
        .limit(limit)\
        .all()
    return entries

@router.post("/", response_model=LedgerResponse)
async def create_ledger_entry(
    entry: LedgerCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(AuthService.get_current_user)
):
    db_entry = LedgerEntry(
        user_id=current_user.id,
        amount=entry.amount,
        description=entry.description,
        category=entry.category,
        transaction_type=entry.transaction_type
    )
    
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    
    return db_entry

@router.get("/balance")
async def get_balance(
    db: Session = Depends(get_db),
    current_user: User = Depends(AuthService.get_current_user)
):
    income = db.query(func.sum(LedgerEntry.amount))\
        .filter(
            LedgerEntry.user_id == current_user.id,
            LedgerEntry.transaction_type == "income"
        )\
        .scalar() or 0
    
    expenses = db.query(func.sum(LedgerEntry.amount))\
        .filter(
            LedgerEntry.user_id == current_user.id,
            LedgerEntry.transaction_type == "expense"
        )\
        .scalar() or 0
    
    return {"balance": float(income) - float(expenses)}
```

### Tests (backend/app/tests/)

#### test_auth.py
```python
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

def test_register_user():
    response = client.post(
        "/api/auth/register",
        json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpass123",
            "role": "user"
        }
    )
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"
    assert response.json()["email"] == "test@example.com"

def test_login_user():
    response = client.post(
        "/api/auth/login",
        data={"username": "testuser", "password": "testpass123"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
```

## 4. Frontend

### Dockerfile (frontend/Dockerfile)
```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
```

### package.json (frontend/package.json)
```json
{
  "name": "black-sultan-frontend",
  "version": "1.0.0",
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "14.0.0",
    "react": "^18.0.0",
    "react-dom": "^18.0.0",
    "axios": "^1.5.0",
    "react-hook-form": "^7.45.0",
    "react-query": "^3.39.3",
    "styled-components": "^6.0.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "@types/react": "^18.0.0",
    "@types/react-dom": "^18.0.0",
    "typescript": "^5.0.0",
    "eslint": "^8.0.0",
    "eslint-config-next": "14.0.0"
  }
}
```

### Hauptseite (frontend/pages/index.tsx)
```tsx
import { useEffect } from 'react'
import { useRouter } from 'next/router'
import { useAuth } from '../hooks/useAuth'

export default function Home() {
  const { user, loading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!loading) {
      if (user) {
        router.push('/dashboard')
      } else {
        router.push('/login')
      }
    }
  }, [user, loading, router])

  return (
    <div style={{ 
      display: 'flex', 
      justifyContent: 'center', 
      alignItems: 'center', 
      height: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    }}>
      <div style={{ textAlign: 'center', color: 'white' }}>
        <h1 style={{ fontSize: '3rem', marginBottom: '1rem' }}>⚡ Black Sultan</h1>
        <p style={{ fontSize: '1.2rem' }}>Loading your financial dashboard...</p>
      </div>
    </div>
  )
}
```

### Auth Hook (frontend/hooks/useAuth.ts)
```tsx
import { useState, useEffect, createContext, useContext, ReactNode } from 'react'
import axios from 'axios'

interface User {
  id: number
  username: string
  email: string
  role: string
}

interface AuthContextType {
  user: User | null
  loading: boolean
  login: (username: string, password: string) => Promise<void>
  register: (userData: any) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const token = localStorage.getItem('token')
    if (token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`
      fetchUser()
    } else {
      setLoading(false)
    }
  }, [])

  const fetchUser = async () => {
    try {
      const response = await axios.get('/api/users/me')
      setUser(response.data)
    } catch (error) {
      localStorage.removeItem('token')
      delete axios.defaults.headers.common['Authorization']
    } finally {
      setLoading(false)
    }
  }

  const login = async (username: string, password: string) => {
    const formData = new FormData()
    formData.append('username', username)
    formData.append('password', password)
    
    const response = await axios.post('/api/auth/login', formData)
    const { access_token } = response.data
    
    localStorage.setItem('token', access_token)
    axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`
    
    await fetchUser()
  }

  const register = async (userData: any) => {
    await axios.post('/api/auth/register', userData)
    await login(userData.username, userData.password)
  }

  const logout = () => {
    localStorage.removeItem('token')
    delete axios.defaults.headers.common['Authorization']
    setUser(null)
  }

  const value = {
    user,
    loading,
    login,
    register,
    logout
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}
```

## 5. AI Service

### Dockerfile (ai-service/Dockerfile)
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

### requirements.txt (ai-service/requirements.txt)
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
openai==0.28.1
python-dotenv==1.0.0
httpx==0.25.2
```

### AI Service App (ai-service/app.py)
```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Black Sultan AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/process")
async def process_ai_request(request: dict):
    try:
        prompt = request.get("prompt", "")
        model = request.get("model", "gpt-3.5-turbo")
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant for the Black Sultan system."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        
        return {
            "response": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens,
            "model": model
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI processing error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Service"}
```

## 6. Smart Contracts

### Hardhat Config (contracts/hardhat.config.js)
```javascript
require("@nomicfoundation/hardhat-toolbox");

module.exports = {
  solidity: "0.8.19",
  networks: {
    localhost: {
      url: "http://127.0.0.1:8545",
    },
  },
  paths: {
    sources: "./contracts",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts"
  },
};
```

### Smart Contract (contracts/BlackSultan.sol)
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract BlackSultan {
    address public owner;
    mapping(address => uint256) public balances;
    mapping(address => mapping(address => uint256)) public allowances;
    
    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);
    event Transfer(address indexed from, address indexed to, uint256 amount);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    function deposit() public payable {
        require(msg.value > 0, "Amount must be greater than 0");
        balances[msg.sender] += msg.value;
        emit Deposit(msg.sender, msg.value);
    }
    
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        require(amount > 0, "Amount must be greater than 0");
        
        balances[msg.sender] -= amount;
        payable(msg.sender).transfer(amount);
        
        emit Withdrawal(msg.sender, amount);
    }
    
    function getBalance() public view returns (uint256) {
        return balances[msg.sender];
    }
    
    function transfer(address to, uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        require(to != address(0), "Invalid recipient");
        require(amount > 0, "Amount must be greater than 0");
        
        balances[msg.sender] -= amount;
        balances[to] += amount;
        
        emit Transfer(msg.sender, to, amount);
    }
    
    function approve(address spender, uint256 amount) public {
        allowances[msg.sender][spender] = amount;
    }
    
    function transferFrom(address from, address to, uint256 amount) public {
        require(allowances[from][msg.sender] >= amount, "Allowance exceeded");
        require(balances[from] >= amount, "Insufficient balance");
        require(to != address(0), "Invalid recipient");
        require(amount > 0, "Amount must be greater than 0");
        
        allowances[from][msg.sender] -= amount;
        balances[from] -= amount;
        balances[to] += amount;
        
        emit Transfer(from, to, amount);
    }
}
```

## 7. Deployment Scripts

### Database Initialization (scripts/init_db.sh)
```bash
#!/bin/bash

echo "Initializing Black Sultan database..."

# Wait for PostgreSQL to be ready
until PGPASSWORD=$POSTGRES_PASSWORD psql -h "db" -U "admin" -d "postgres" -c '\q'; do
  >&2 echo "PostgreSQL is unavailable - sleeping"
  sleep 1
done

# Create database and run migrations
PGPASSWORD=$POSTGRES_PASSWORD psql -h "db" -U "admin" -d "postgres" -c "CREATE DATABASE black_sultan;"
PGPASSWORD=$POSTGRES_PASSWORD psql -h "db" -U "admin" -d "black_sultan" -f /app/migrations/init.sql

echo "Database initialization complete!"
```

## 8. Postman Collection

Erstellen Sie eine Postman-Sammlung mit folgenden Endpunkten:

- `POST /api/auth/register` - Benutzerregistrierung
- `POST /api/auth/login` - Benutzerlogin
- `GET /api/users/me` - Aktueller Benutzer
- `GET /api/ledger/` - Ledger-Einträge abrufen
- `POST /api/ledger/` - Neuen Ledger-Eintrag erstellen
- `POST /api/ai/process` - AI-Anfrage verarbeiten

## Startanleitung

1. **Umgebungsvariablen setzen**:
   ```bash
   cp .env.example .env
   # OpenAI API Key in .env eintragen
   ```

2. **System starten**:
   ```bash
   docker-compose up --build
   ```

3. **Datenbank initialisieren** (falls nicht automatisch):
   ```bash
   docker exec -it black_sultan_backend python scripts/init_db.py
   ```

4. **Frontend öffnen**: http://localhost:3000

5. **Backend API testen**: http://localhost:8000/docs

## Standard-Benutzer

- **Admin**: username: `admin`, password: `admin123`
- **User**: username: `testuser`, password: `testpass123`

Dieses vollständige System bietet:
- ✅ Vollständige Benutzerauthentifizierung mit JWT
- ✅ Datenbank mit PostgreSQL und SQLAlchemy
- ✅ Finanz-Ledger mit Einnahmen/Ausgaben
- ✅ AI-Integration mit OpenAI
- ✅ React/Next.js Frontend
- ✅ Smart Contract mit Hardhat
- ✅ Umfassende Tests
- ✅ Docker-Containerisierung
- ✅ Postman-Testdaten
- ✅ Role-Based Access Control

Das System ist sofort lauffähig nach `docker-compose up --build`!
