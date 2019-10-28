<template>
	<v-app style="background-color: #171038">
		<v-content>
			<v-toolbar>
				<v-app-bar-nav-icon>
					<v-img src="@/assets/img/up-and-down/favicon-32x32.png"></v-img>
				</v-app-bar-nav-icon>
				<v-toolbar-title>UP & DOWN</v-toolbar-title>

				<v-spacer></v-spacer>

				<v-toolbar-items>
					<v-btn text>Link 1</v-btn>
					<v-btn text>Link 2</v-btn>
					<v-btn text>Link 3</v-btn>
				</v-toolbar-items>
			</v-toolbar>
			<v-flex id="stock-chart-container" xs12>
				<stock-chart-view></stock-chart-view>
			</v-flex>
			<v-flex id="info-chart-container" xs12>
				test
			</v-flex>
			<team-info></team-info>
		</v-content>
	</v-app>
</template>

<script>
	import StockChartView from "@/Views/StockChartView";
	import TeamInfo from "./components/TeamInfo";

	export default {
		name: 'App',
		components: {
			TeamInfo,
			StockChartView,
		},
		data: () => ({
		}),
		mounted() {
		}
	};
</script>

<style>
	* {
		box-sizing: border-box;
	}

	#stock-chart-container {
	}

	#info-chart-container {
	}
</style>