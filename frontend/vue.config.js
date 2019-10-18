module.exports = {
    "outputDir": "../backend/dist/",
    "assetsDir": "assets",
    "transpileDependencies": [
        "vuetify"
    ],
    devServer: {
        proxy: {
            '/api': {
                target: 'http://localhost:8000',
                changeOrigin: true
            }
        },
        headers: {
            "Access-Control-Allow-Origin": "*"
        },
    }
};
