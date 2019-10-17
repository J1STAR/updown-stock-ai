<template>
	<v-container fluid fill-height pa-0>
		<v-layout
				align-center
				wrap
		>
			<v-flex xs10>
				<GChart
						type="LineChart"
						:data="chartData"
						:options="chartOptions"
						:events="chartEvents"
				/>
			</v-flex>
			<v-flex xs2>
				predict container
			</v-flex>
		</v-layout>
	</v-container>
</template>

<script>
	export default {
		name: "LineChart",
		props: {},
		data() {
			return {
				chartData: [
					['Year', 'Sales'],
					['2019/10/01', 1000],
					['2019/10/02', 1170],
					['2019/10/03', 660],
					['2019/10/04', 1030],
					['2019/10/05', 660],
					['2019/10/06', 660],
					['2019/10/07', 660],
					['2019/10/08', 660],
					['2019/10/09', 660],
					['2019/10/10', 660],
					['2019/10/11', 660],
					['2019/10/12', 660],
					['2019/10/13', 660],
					['2019/10/14', 660],
					['2019/10/15', 660],
				],
				chartOptions: {
					chart: {
						title: 'Company Performance',
						subtitle: 'Sales, Expenses, and Profit: 2014-2017',
					},
					pointsVisible: true,
					height: 800,
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
			renderPredict: function() {
				let stockChart = document.querySelector('svg');

				let circles = document.querySelectorAll('svg g circle');
				let lastCircle = circles[circles.length - 1];

				let cx = Number(lastCircle.getAttribute('cx'));
				let cy = Number(lastCircle.getAttribute('cy'));

				stockChart.innerHTML += '<defs><marker id="Triangle" fill="red" stroke="red" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z"></path></marker></defs><polyline points="'+cx+','+cy+' '+(cx+50)+','+(cy-50)+'" fill="none" stroke="red" stroke-width="2" marker-end="url(#Triangle)"></polyline><polyline points="'+cx+','+cy+' '+(cx+50)+','+(cy-100)+'" fill="none" stroke="red" stroke-width="2" marker-end="url(#Triangle)"></polyline><polyline points="'+cx+','+cy+' '+(cx+50)+','+(cy+50)+'" fill="none" stroke="red" stroke-width="2" marker-end="url(#Triangle)"></polyline>'
			}
		},
		filters: {},
		components: {}
	}
</script>

<style scoped>

</style>