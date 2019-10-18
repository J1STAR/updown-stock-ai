<template>
	<v-app>
		<v-content>
			<v-flex id="stock-chart-container" xs12>
				<stock-chart-view></stock-chart-view>
			</v-flex>
			<v-flex id="info-chart-container" xs12>

			</v-flex>
		</v-content>
	</v-app>
</template>

<script>
	import StockChartView from "@/Views/StockChartView";

	export default {
		name: 'App',
		components: {
			StockChartView,
		},
		data: () => ({
		}),
	};
</script>

<style>
	* {
		box-sizing: border-box;
	}

	body {
		background-color: white;
	}

	#stock-chart-container {
		height: 600px
	}

	#info-chart-container {
		height: 300px
	}
</style>