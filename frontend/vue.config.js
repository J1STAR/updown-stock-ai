module.exports = {
  "outputDir": "../backend/templates/",
  "assetsDir": "../static/assets",
  "transpileDependencies": [
    "vuetify"
  ],
  devServer: {
    headers: {
      "Access-Control-Allow-Origin": "*"
    },
  }
};
