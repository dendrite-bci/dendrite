import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [vue(), tailwindcss()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8321',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8321',
        ws: true,
      },
    },
  },
})
