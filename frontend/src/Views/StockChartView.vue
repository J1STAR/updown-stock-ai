<template>
	<v-container fluid id="stock-chart-container">
		<v-layout id="stock-info-layout" wrap>
			<v-flex
					id="stock-info-selector-container"
					pa-4
					xs12
					sm12
					md2
			>
				<v-autocomplete
						label="업종"
						dark
						v-model="businessType"
						@change="loadCorporations"
						item-text="name"
						:items="businessTypes"
						return-object
				></v-autocomplete>
				<v-autocomplete
						label="종목"
						dark
						v-model="corp"
						@change="loadCorporationInfo"
						item-text="name"
						:items="corporations"
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
								dark
								v-model="date1"
								label="StartDate"
								prepend-icon="mdi-calendar"
								readonly
								v-on="on"
						></v-text-field>
					</template>
					<v-date-picker v-model="date1" no-title scrollable style="z-index:999">
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
								dark
								v-model="date2"
								label="EndDate"
								prepend-icon="mdi-calendar"
								readonly
								v-on="on"
						></v-text-field>
					</template>
					<v-date-picker v-model="date2" no-title scrollable style="z-index:999">
						<v-spacer></v-spacer>
						<v-btn text color="primary" @click="menu2 = false">Cancel</v-btn>
						<v-btn text color="primary" @click="$refs.menu2.save(date2)">OK</v-btn>
					</v-date-picker>
				</v-menu>
			</v-flex>
			<v-flex
					xs12
					sm12
					md8
			>
				<candle-chart :date1="date1" :date2="date2" :chartData="chartData" :isActive="status" :title="corp.name"/>
			</v-flex>
			<v-flex
					id="stock-info-additional-info-container"
					pa-4
					xs12
					sm12
					md2
			>
				<v-timeline dense>
					<v-timeline-item :color="'red'" :icon="'mdi-arrow-top-right-bold-outline'">
						<v-card class="elevation-2">
							<v-card-text>D+1 주가 상승 예측</v-card-text>
						</v-card>
					</v-timeline-item>
					<v-timeline-item :icon="'mdi-arrow-bottom-right-bold-outline'">
						<v-card class="elevation-2">
							<v-card-text>D+2 주가 하락 예측</v-card-text>
						</v-card>
					</v-timeline-item>
					<v-timeline-item :color="'red'" :icon="'mdi-arrow-top-right-bold-outline'">
						<v-card class="elevation-2">
							<v-card-text>D+3 주가 상승 예측</v-card-text>
						</v-card>
					</v-timeline-item>
					<v-timeline-item :color="'red'" :icon="'mdi-arrow-top-right-bold-outline'">
						<v-card class="elevation-2">
							<v-card-text>D+4 주가 상승 예측</v-card-text>
						</v-card>
					</v-timeline-item>
					<v-timeline-item :icon="'mdi-arrow-bottom-right-bold-outline'">
						<v-card class="elevation-2">
							<v-card-text>D+5 주가 하락 예측</v-card-text>
						</v-card>
					</v-timeline-item>
					<v-timeline-item :icon="'mdi-arrow-bottom-right-bold-outline'">
						<v-card class="elevation-2">
							<v-card-text>D+6 주가 하락 예측</v-card-text>
						</v-card>
					</v-timeline-item>
					<v-timeline-item :icon="'mdi-arrow-bottom-right-bold-outline'">
						<v-card class="elevation-2">
							<v-card-text>D+7 주가 하락 예측</v-card-text>
						</v-card>
					</v-timeline-item>
				</v-timeline>
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
				corporations: [
				],
				chartData: [],
			};
		},
		async mounted() {
			await this.$store.dispatch('stock/loadBusinessTypes')
			this.businessTypes = this.$store.getters['stock/getBusinessTypes']
			this.businessType = this.businessTypes[0]

			await this.loadCorporations()
			let startDate = new Date()
			startDate.setDate(startDate.getDate() - 30)
			this.date1 = startDate.toISOString().substr(0, 10)
		},
		methods: {
			loadCorporations: async function() {
				await this.$store.dispatch('stock/loadCorporations', this.businessType.business_code)
				this.corporations = this.$store.getters['stock/getCorporations']
				this.corp = this.corporations[0]

				this.loadCorporationInfo()
			},
			loadCorporationInfo: async function() {
				this.status = false

				let res = await this.$http.get("/stock/corp/"+ this.corp.corp_code)

				let stock_info = res.data.corp.stock_info

				if(this.date1 >= this.date2) {
					this.date2 = this.date1
				}

				this.chartData = []
				for(let row of stock_info) {
					let currentDate = new Date(row['date'].substr(0, 10))

					if(row['open_price'] !== 0) {
						this.chartData.push([currentDate.getTime(), row['open_price'], row['high_price'], row['low_price'], row['closing_price'], row['volume']])
					}
				}

				this.$emit("changeCorp", this.corp)
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
	#stock-chart-container {
		min-height: 600px
	}

	#stock-info-layout {
		min-height: 600px;

		box-shadow: whitesmoke 0px 2px 12px 2px
	}

	#stock-info-layout #stock-info-selector-container, #stock-info-additional-info-container {
		background-color: #231C43
	}
</style>