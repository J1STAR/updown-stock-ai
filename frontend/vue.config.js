module.exports = {
  "outputDir": "../backend/dist/",
  "assetsDir": "assets",
  "transpileDependencies": [
    "vuetify"
  ],
  devServer: {
    proxy: {
      "/stock": {
        target: "http://j1star.ddns.net:8000",
        changeOrigin: true
      },
      "/news": {
        target: "http://j1star.ddns.net:8000",
        changeOrigin: true
      }
    },
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': '*',
    },
    disableHostCheck: true,
    host: "0.0.0.0"
  }
};
