module.exports = {
  "outputDir": "../backend/dist/",
  "assetsDir": "assets",
  "transpileDependencies": [
    "vuetify"
  ],
  devServer: {
    proxy: {
      "/item": {
        target: "https://finance.naver.com",
        changeOrigin: true
      }
    },
    headers: {
      "Access-Control-Allow-Origin": "*"
    },
  }
};
