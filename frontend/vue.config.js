module.exports = {
  "outputDir": "../backend/dist/",
  "assetsDir": "assets",
  "transpileDependencies": [
    "vuetify"
  ],
  devServer: {
    headers: {
      "Access-Control-Allow-Origin": "*"
    },
  }
};
