SharkByte AI Demo Frontend

This project is a React-based frontend application built using Vite and Tailwind CSS. The frontend communicates with a backend server through API and WebSocket connections. The project includes responsive styling, custom themes, and development server proxy configuration for local testing.

Technologies Used
React
Vite
Tailwind CSS
PostCSS
Framer Motion
Lucide React
QRCode React

The project dependencies and development scripts are defined in package.json.

Project Structure
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
File Descriptions
package.json

The package.json file manages the project configuration, dependencies, and scripts. It defines the React, Vite, and Tailwind CSS packages used by the application. It also includes commands for running and building the project.

Important Scripts
npm run dev

Starts the development server.

npm run build

Builds the project for production.

npm run preview

Previews the production build locally.

package-lock.json

The package-lock.json file automatically stores the exact versions of all installed dependencies. This ensures consistent installations across different development environments and systems.

postcss.config.js

This file configures PostCSS plugins used in the project. It enables Tailwind CSS and Autoprefixer support for processing CSS files.

Configured Plugins
Tailwind CSS
Autoprefixer
tailwind.config.js

This file customizes the Tailwind CSS framework used in the application. It defines the content scanning paths, custom fonts, colors, and box shadows for the user interface.

Custom Theme Features
Custom color palette
Rounded display fonts
Custom shadow effects

Example custom colors:

foam
deep
lagoon
coral
vite.config.js

This file configures the Vite development server and React integration. It also sets up API and WebSocket proxy routing to connect the frontend with a backend server running on port 8000.

Features
React plugin support
API request proxying
WebSocket proxy support
Backend error handling
Local development stability improvements
Proxy Configuration
/api
Routes API requests to http://localhost:8000
/ws
Routes WebSocket connections to ws://localhost:8000
Installation
1. Install Dependencies
npm install
Running the Project
Start Development Server
npm run dev

The application will start on the Vite development server.

Build for Production
npm run build
Preview Production Build
npm run preview
Notes
The frontend expects a backend server running on port 8000.
API requests are automatically proxied through Vite.
WebSocket support is enabled for real-time communication features.
Tailwind CSS is used for styling and responsive design.
