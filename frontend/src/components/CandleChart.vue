<template>
    <v-container fluid fill-height pa-0>
        <v-layout
                justify-center
                align-center
                wrap
                id="chartLayout"
        >
            <v-flex xs12 text-center>
                <template v-if="isActive">
                    <trading-vue ref="tradingVue" :data="chart"
                                 :title-txt="title"
                                 :width="width"
                                 :height="height"
                                 :color-back="colors.colorBack"
                                 :color-grid="colors.colorGrid"
                                 :color-text="colors.colorText"
                                 :color-candle-up="colors.colorCandleUp"
                                 :color-candle-dw="colors.colorCandleDw">
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
                    colorCandleUp: 'blue',
                    colorCandleDw: 'red',
                }
            }
        },
        watch: {
            chartData: function (newVal) {
                this.chart.ohlcv = newVal
                this.width = document.querySelector('#chartLayout').offsetWidth
                this.height = document.querySelector('#chartLayout').offsetHeight

                this.$nextTick(() =>
                    this.$refs.tradingVue.setRange(this.chart.ohlcv[0][0], this.chart.ohlcv[this.chart.ohlcv.length - 1][0])
                )
            }
        },
        mounted() {

        },
        methods: {},
        filters: {},
        components: {
            TradingVue
        }
    }
</script>

<style scoped>

</style>