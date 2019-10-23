<template>
	<v-container fluid fill-height pa-0>
		<v-layout
				align-center
				wrap
		>
			<v-flex xs12 text-center>
				<template v-if="chartData.length != 0">
					<GChart
							type="LineChart"
							:data="chartData"
							:options="chartOptions"
							:events="chartEvents"
					/>
				</template>
				<template v-else>
					<v-progress-circular
							:size="256"
							:width="10"
							color="purple"
							indeterminate
					></v-progress-circular>
				</template>
			</v-flex>
		</v-layout>
	</v-container>
</template>

<script>
	export default {
		name: "LineChart",
		props: {
			chartData: {
				type: Array,
			}
		},
		data() {
			return {
				chartOptions: {
					chart: {
						title: 'Company Performance',
						subtitle: 'Sales, Expenses, and Profit: 2014-2017',
					},
					title: "일별 시세 차트",
					titleTextStyle: {
						fontSize: 64
					},
					height: 600 - 24,
					pointsVisible: true,
					series: {
						0: {visibleInLegend: false}
					},
					vAxis: {
						title: "Closing Price",
						format: "# 원"
					},
					hAxis: {
						title: "Date",
						slantedText: true,
						slantedTextAngle: 90
					}
				},
				chartEvents: {
					'ready': () => {
						this.renderPredict();
					}
				}
			}
		},
		mounted() {
		},
		methods: {
			renderPredict: function () {
				let stockChart = document.querySelector('svg > g:nth-child(3)');

				let circles = document.querySelectorAll('svg g circle');
				let lastCircle = circles[circles.length - 1];

				let cx = Number(lastCircle.getAttribute('cx'));
				let cy = Number(lastCircle.getAttribute('cy'));

				stockChart.innerHTML = '<g><defs><marker id="Triangle" fill="red" stroke="red" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z"></path></marker></defs><polyline points="' + cx + ',' + cy + ' ' + (cx + 50) + ',' + (cy - 50) + '" fill="none" stroke="red" stroke-width="2" marker-end="url(#Triangle)"></polyline><polyline points="' + cx + ',' + cy + ' ' + (cx + 50) + ',' + (cy - 100) + '" fill="none" stroke="red" stroke-width="2" marker-end="url(#Triangle)"></polyline><polyline points="' + cx + ',' + cy + ' ' + (cx + 50) + ',' + (cy + 50) + '" fill="none" stroke="red" stroke-width="2" marker-end="url(#Triangle)"></polyline></g>' + stockChart.innerHTML
			}
		},
		filters: {},
		components: {}
	}
</script>

<style scoped>

</style>