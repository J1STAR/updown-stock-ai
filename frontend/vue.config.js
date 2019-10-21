module.exports = {
  "outputDir": "../backend/dist/",
  "assetsDir": "assets",
  "transpileDependencies": [
    "vuetify"
  ],
  devServer: {
    proxy: {
      "/stock": {
        target: "http://localhost:8000",
        changeOrigin: true
      }
    },
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': '*',
    },
  }
};
