// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // 👈 This is critical to allow external access (0.0.0.0)
    port: 5173, // (optional) define the port explicitly
  },
});