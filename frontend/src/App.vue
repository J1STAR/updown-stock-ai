<template>
	<v-app style="background-color: #171038">
		<v-content>
			<v-toolbar>
				<v-app-bar-nav-icon>
					<v-img src="@/assets/img/up-and-down/favicon-32x32.png"></v-img>
				</v-app-bar-nav-icon>
				<v-toolbar-title>UP & DOWN</v-toolbar-title>

				<v-spacer></v-spacer>

				<!--<v-toolbar-items>
					<v-btn text>Link 1</v-btn>
					<v-btn text>Link 2</v-btn>
					<v-btn text>Link 3</v-btn>
				</v-toolbar-items>-->
			</v-toolbar>
			<v-flex xs12>
				<stock-chart-view @changeCorp="setCorp"></stock-chart-view>
			</v-flex>
			<v-flex xs12>
				<news-list :news_target="corp"></news-list>
			</v-flex>
			<team-info></team-info>
		</v-content>
	</v-app>
</template>

<script>
	import StockChartView from "@/Views/StockChartView";
	import TeamInfo from "./components/TeamInfo";
	import NewsList from "./components/NewsList";

	export default {
		name: 'App',
		components: {
			NewsList,
			TeamInfo,
			StockChartView,
		},
		data: () => ({
			corp: {}
		}),
		mounted() {
		},
		methods: {
			setCorp: function(corp) {
				this.corp = corp
			}
		}
	};
</script>

<style>
	* {
		box-sizing: border-box;
	}
</style>