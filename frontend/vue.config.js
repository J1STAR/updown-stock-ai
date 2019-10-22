module.exports = {
    "outputDir": "../backend/dist/",
    "assetsDir": "assets",
    "transpileDependencies": [
        "vuetify"
    ],
    devServer: {
        proxy: {
            '/api': {
                target: 'http://localhost:8000', // 8000 포트로 리다이렉트한다.
                changeOrigin: true
            }
        },
        headers: {
            "Access-Control-Allow-Origin": "*"
        },
    }
};
