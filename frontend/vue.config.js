module.exports = {
<<<<<<< HEAD
  "outputDir": "../backend/dist/",
  "assetsDir": "assets",
  "transpileDependencies": [
    "vuetify"
  ],
  devServer: {
    proxy: {
      "/api": {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    },
    headers: {
      "Access-Control-Allow-Origin": "*"
    },
  }
=======
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
>>>>>>> b17d8142c95937e94a2d3babc9ae1e181213a81a
};
