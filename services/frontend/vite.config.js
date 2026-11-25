import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    server: {
        host: true, // 讓 Docker 外部可以存取
        port: 5173,
        // Windows/macOS 掛載進 Docker 的資料夾收不到檔案變更通知，改用輪詢才能自動重新載入
        watch: { usePolling: true, interval: 500 }
    }
})