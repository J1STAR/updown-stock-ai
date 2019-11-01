<template>
    <v-container id="candle-chart-container" fill-height fluid pa-0>
        <v-layout
                justify-center
                align-center
                wrap
                id="chartLayout"
        >
            <v-flex sm12 text-center>
                <template v-if="isActive">
                    <trading-vue ref="tradingVue" :data="chart"
                                 :title-txt="title"
                                 :width="width"
                                 :height="height"
                                 :color-back="colors.colorBack"
                                 :color-grid="colors.colorGrid"
                                 :color-text="colors.colorText"
                                 :color-candle-up="colors.colorCandleUp"
                                 :color-candle-dw="colors.colorCandleDw"
                                 :color-vol-up="colors.colorCandleUp"
                                 :color-vol-dw="colors.colorCandleDw"
                                 :color-wick-up="colors.colorCandleUp"
                                 :color-wick-dw="colors.colorCandleDw"
                    >
                    </trading-vue>
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
    import TradingVue from 'trading-vue-js'

    export default {
        name: "CandleChart",
        props: {
            title: {
                type: String
            },
            isActive: {
                type: Boolean,
                default: false
            },
            chartData: {
                type: Array
            },
            date1: {
                type: String
            },
            date2: {
                type: String
            }
        },
        data() {
            return {
                chart: {
                    ohlcv: [],
                },
                width: 0,
                height: 0,
                colors: {
                    colorBack: '#fff',
                    colorGrid: '#eee',
                    colorText: '#333',
                    colorCandleUp: 'red',
                    colorCandleDw: 'blue',
                }
            }
        },
        watch: {
            date1: function () {
                if(this.chartData.length !== 0) {
                    this.drawChart()
                }
            },
            date2: function () {
                if(this.chartData.length !== 0) {
                    this.drawChart()
                }
            },
            chartData: function () {
                this.setChartData()
            },
        },
        mounted() {
            window.addEventListener('resize', this.resizeChart)
        },
        methods: {
            resizeChart: function() {
                this.width = document.querySelector('#chartLayout').offsetWidth
                this.height = document.querySelector('#chartLayout').offsetHeight
            },
            drawChart: function() {
                let startDate = new Date(this.date1)
                startDate.setDate(startDate.getDate() - 1)

                let endDate = new Date(this.date2)
                endDate.setDate(endDate.getDate() + 1)

                this.$nextTick(() =>
                    this.$refs.tradingVue.setRange(Number(startDate), Number(endDate))
                )
                this.resizeChart()
            },
            setChartData: function() {
                this.chart.ohlcv = this.chartData
                this.drawChart()
            }
        },
        filters: {},
        components: {
            TradingVue
        }
    }
</script>

<style scoped>
    #candle-chart-container {
        min-height: 600px
    }
</style>