import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on("error", (error, request, response) => {
            if (response?.writeHead && !response.headersSent) {
              response.writeHead(502, { "Content-Type": "application/json" });
              response.end(JSON.stringify({ detail: "Backend unavailable" }));
            }
          });
        },
      },
      "/ws": {
        target: "ws://localhost:8000",
        changeOrigin: true,
        ws: true,
        configure: (proxy) => {
          proxy.on("error", () => {
            //Keep Vite alive when the backend socket is offline.
          });
          proxy.on("proxyReqWs", (proxyRequest) => {
            proxyRequest.on("error", () => {
              //Ignore socket reset errors during local testing.
            });
          });
          proxy.on("open", (proxySocket) => {
            proxySocket.on("error", () => {
              //Ignore backend socket resets after the browser disconnects.
            });
          });
        },
      },
    },
  },
});
