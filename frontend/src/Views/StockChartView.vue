<template>
	<v-container fluid fill-height>
		<v-layout>
			<v-flex
					pa-4
					xs2
			>
				<v-autocomplete
						label="업종"
						v-model="businessType"
						@change="reloadCorparations"
						item-text="name"
						:items="businessTypes"
						return-object
				></v-autocomplete>
				<v-autocomplete
						label="종목"
						v-model="corp"
						@change="reloadCorparationInfo"
						item-text="name"
						:items="corparations"
						return-object
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
				<v-btn @click="test">test insert</v-btn>
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
				menu1: false,
				date1: new Date('2019-10-04').toISOString().substr(0, 10),
				menu2: false,
				date2: new Date().toISOString().substr(0, 10),
				businessType: "",
				businessTypes: [
				],
				corp: "",
				corparations: [
				],
			};
		},
		async mounted() {
			await this.$store.dispatch('stock/loadBusinessTypes')
			this.businessTypes = this.$store.getters['stock/getBusinessTypes']
			this.businessType = this.businessTypes[0]

			this.reloadCorparations()
		},
		methods: {
			test: function () {
				this.$http.post('/stock/005931/', {
					"corp_name": "삼성전자",
					"stock_info": [
						{
							"date": "2019.10.22",
							"closing_price": 47800,
							"diff": -7899,
							"open_price": 567868,
							"high_price": 6786,
							"low_price": 56783,
							"volumn": 37435
						},
					]
				})
				.then((res) => {
					console.log(res)
				})
			},
			reloadCorparations: async function() {
				await this.$store.dispatch('stock/loadCorparations', this.businessType.business_code)
				this.corparations = this.$store.getters['stock/getCorparations']
				this.corp = this.corparations[0]
			},
			reloadCorparationInfo: function() {

			}
		},
		filters: {},
		components: {LineChart}
	}
</script>

<style scoped>

</style>