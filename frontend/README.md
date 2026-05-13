# SharkByte AI Demo Frontend

A React-based frontend application built using Vite and Tailwind CSS. The frontend communicates with a backend server through API and WebSocket connections while supporting responsive styling and real-time communication.

---

## Technologies Used

- React
- Vite
- Tailwind CSS
- PostCSS
- Framer Motion
- Lucide React
- QRCode React

---

## Project Structure

```plaintext
frontend/
│
├── src/
│
├── README.md
├── package.json
├── package-lock.json
├── postcss.config.js
├── tailwind.config.js
└── vite.config.js
```

---

## File Descriptions

### package.json

The `package.json` file manages the project configuration, dependencies, and scripts. It defines the React, Vite, and Tailwind CSS packages used by the application.

#### Important Scripts

```bash
npm run dev
```

Starts the development server.

```bash
npm run build
```

Builds the project for production.

```bash
npm run preview
```

Previews the production build locally.

---

### package-lock.json

The `package-lock.json` file stores the exact versions of installed dependencies to ensure consistent installations across different systems.

---

### postcss.config.js

This file configures PostCSS plugins used in the project.

#### Configured Plugins

- Tailwind CSS
- Autoprefixer

---

### tailwind.config.js

This file customizes the Tailwind CSS framework used in the application.

#### Features

- Custom fonts
- Custom color palette
- Custom shadow effects
- Responsive styling support

#### Example Custom Colors

- foam
- deep
- lagoon
- coral

---

### vite.config.js

This file configures the Vite development server and React integration.

#### Features

- React plugin support
- API proxy routing
- WebSocket proxy support
- Backend error handling
- Local development stability improvements

#### Proxy Configuration

- `/api`
  - Routes API requests to `http://localhost:8000`

- `/ws`
  - Routes WebSocket connections to `ws://localhost:8000`

---

## Installation

Install all required dependencies:

```bash
npm install
```

---

## Running the Project

Start the development server:

```bash
npm run dev
```

---

## Build for Production

```bash
npm run build
```

---

## Preview Production Build

```bash
npm run preview
```

---

## Notes

- The frontend expects a backend server running on port `8000`.
- API requests are automatically proxied through Vite.
- WebSocket support is enabled for real-time communication.
- Tailwind CSS is used for styling and responsive design.
