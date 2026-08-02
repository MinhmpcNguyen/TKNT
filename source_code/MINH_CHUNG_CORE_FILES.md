## Quick Installation Guide

### Requirements

- Python 3.11+
- Node.js 18+
- Docker and Docker Compose

### Quick Start With Docker

```bash
cp .env.example .env
docker compose up --build
```

Open `http://localhost:8080`. The API initializes the runtime database tables automatically on startup.

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn api.main:app --reload
```

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, usually `http://localhost:5173`.

Note for local development: PostgreSQL with pgvector must be running, and the database connection variables in `.env` must point to the correct database. For the simplest setup, use Docker Compose.

