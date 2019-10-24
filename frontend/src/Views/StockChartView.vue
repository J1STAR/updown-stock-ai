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
								@keydown="test"
								v-on="on"
						></v-text-field>
					</template>
					<v-date-picker v-model="date1" @change="reloadCorparationInfo" no-title scrollable>
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
					<v-date-picker v-model="date2" @change="reloadCorparationInfo" no-title scrollable>
						<v-spacer></v-spacer>
						<v-btn text color="primary" @click="menu2 = false">Cancel</v-btn>
						<v-btn text color="primary" @click="$refs.menu2.save(date2)">OK</v-btn>
					</v-date-picker>
				</v-menu>
			</v-flex>
			<v-flex
					xs8
			>
				<candle-chart :chartData="chartData" :isActive="status" :title="corp.name"/>
			</v-flex>
			<v-flex
					xs2
			>

			</v-flex>
		</v-layout>
	</v-container>
</template>

<script>
	import CandleChart from "@/components/CandleChart";

	export default {
		name: "StockChartView",
		props: {},
		data() {
			return {
				status: false,
				menu1: false,
				date1: new Date().toISOString().substr(0, 10),
				menu2: false,
				date2: new Date().toISOString().substr(0, 10),
				businessType: "",
				businessTypes: [
				],
				corp: "",
				corparations: [
				],
				chartData: [],
			};
		},
		async mounted() {
			await this.$store.dispatch('stock/loadBusinessTypes')
			this.businessTypes = this.$store.getters['stock/getBusinessTypes']
			this.businessType = this.businessTypes[0]

			this.reloadCorparations()

			let startDate = new Date()
			startDate.setDate(startDate.getDate() - 30)
			this.date1 = startDate.toISOString().substr(0, 10)
		},
		methods: {
			test: function() {
				console.log("test")
			},
			reloadCorparations: async function() {
				await this.$store.dispatch('stock/loadCorparations', this.businessType.business_code)
				this.corparations = this.$store.getters['stock/getCorparations']
				this.corp = this.corparations[0]

				this.reloadCorparationInfo()
			},
			reloadCorparationInfo: async function() {
				this.status = false

				let res = await this.$http.get("/stock/"+ this.corp.corp_code)

				let stock_info = res.data.corp.stock_info

				if(this.date1 >= this.date2) {
					this.date2 = this.date1
				}

				this.chartData = []
				for(let row of stock_info) {
					let startDate = new Date(this.date1)
					let currentDate = new Date(row['date'].substr(0, 10))
					let endDate = new Date(this.date2)
					if(startDate <= currentDate && currentDate <= endDate) {
						this.chartData.push([currentDate.getTime(), row['open_price'], row['high_price'], row['low_price'], row['closing_price'], row['volumn']])
					}
				}

				this.status = true
			}
		},
		filters: {},
		components: {
			CandleChart
		}
	}
</script>

<style scoped>

</style>