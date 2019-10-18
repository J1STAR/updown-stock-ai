<template>
	<v-container fluid fill-height>
		<v-layout>
			<v-flex
					pa-4
					xs2
			>
				<v-autocomplete
						label="Corparations"
						:value="corp"
						:items="corparations"
				></v-autocomplete>
				<v-menu
						ref="menu1"
						v-model="menu1"
						:close-on-content-click="false"
						:return-value.sync="date1"
						transition="scale-transition"
						offset-y
						min-width="290px"
				>
					<template v-slot:activator="{ on }">
						<v-text-field
								v-model="date1"
								label="StartDate"
								prepend-icon="mdi-calendar"
								readonly
								v-on="on"
						></v-text-field>
					</template>
					<v-date-picker v-model="date1" no-title scrollable>
						<v-spacer></v-spacer>
						<v-btn text color="primary" @click="menu1 = false">Cancel</v-btn>
						<v-btn text color="primary" @click="$refs.menu1.save(date1)">OK</v-btn>
					</v-date-picker>
				</v-menu>
				<v-menu
						ref="menu2"
						v-model="menu2"
						:close-on-content-click="false"
						:return-value.sync="date2"
						transition="scale-transition"
						offset-y
						min-width="290px"
				>
					<template v-slot:activator="{ on }">
						<v-text-field
								v-model="date2"
								label="EndDate"
								prepend-icon="mdi-calendar"
								readonly
								v-on="on"
						></v-text-field>
					</template>
					<v-date-picker v-model="date2" no-title scrollable>
						<v-spacer></v-spacer>
						<v-btn text color="primary" @click="menu2 = false">Cancel</v-btn>
						<v-btn text color="primary" @click="$refs.menu2.save(date2)">OK</v-btn>
					</v-date-picker>
				</v-menu>
			</v-flex>
			<v-flex
					xs8
			>
				<line-chart/>
			</v-flex>
			<v-flex
					xs2
			>
				<v-btn @click="test"></v-btn>
			</v-flex>
		</v-layout>
	</v-container>
</template>

<script>
	import LineChart from "@/components/LineChart";

	export default {
		name: "StockChartView",
		props: {},
		data() {
			return {
				modal: false,
				menu1: false,
				date1: new Date('2019-10-04').toISOString().substr(0, 10),
				menu2: false,
				date2: new Date().toISOString().substr(0, 10),
				corp: "삼성전자",
				corparations: [
					"KB 금융", "KT", "KT&G", "LG", "네이버", "삼성전자"
				],
			};
		},
		mounted() {

		},
		methods: {
			test: function() {
				this.$http.get('/item/sise_day.nhn?code=005930&page=1')
					.then((res) => {
					console.log(res)
				})
			}
		},
		filters: {},
		components: {LineChart}
	}
</script>

<style scoped>

</style>