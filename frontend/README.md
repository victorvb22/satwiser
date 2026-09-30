# Frontend

React + TypeScript + Vite. Screens: Mission, Labo (degradation lab recomputed in a Web
Worker), Méthode.

    npm install
    npm run dev        # http://127.0.0.1:5173, proxies /api to the local API on :8000
    npm test           # parity with the Python lab, worker logic, screen rendering

Production (Vercel): set `VITE_API_URL` to the hosted API origin.
