import {fileURLToPath, URL} from 'node:url'

import {defineConfig} from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
    plugins: [
        vue(),
        vueDevTools(),
        tailwindcss(),
    ],
    resolve: {
        alias: {
            '@': fileURLToPath(new URL('./src', import.meta.url))
        },
    },
    server: {
        proxy: {
            '/accounts': {
                target: 'http://127.0.0.1:8000/accounts',
                changeOrigin: true,
                rewrite: path => path.replace(/^\/accounts/, '')
            },
            '/api': {
                target: 'http://127.0.0.1:8000/api',
                changeOrigin: true,
                rewrite: path => path.replace(/^\/api/, '')
            },
            '/lstm': {
                target: 'http://127.0.0.1:8000/lstm',
                changeOrigin: true,
                rewrite: path => path.replace(/^\/lstm/, '')
            },
            '/auth': {
                target: 'http://127.0.0.1:8000/auth',
                changeOrigin: true,
                rewrite: path => path.replace(/^\/auth/, '')
            }
        }
    }
})
