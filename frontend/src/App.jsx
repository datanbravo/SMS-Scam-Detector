import Dashboard from "./components/Dashboard.jsx";
import SubmitPage from "./components/SubmitPage.jsx";

//This is where the frontend finds the backend.
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

export function getDashboardSocketUrl() {
  const customSocketUrl = import.meta.env.VITE_WS_BASE_URL;

  if (customSocketUrl) {
    return `${customSocketUrl.replace(/\/$/, "")}/ws/dashboard`;
  }

  const socketType = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${socketType}//${window.location.host}/ws/dashboard`;
}

export default function App() {
  const pagePath = window.location.pathname;

  //This keeps the phone submit page separate from the dashboard.
  if (pagePath.startsWith("/submit")) {
    return <SubmitPage />;
  }

  return <Dashboard />;
}
