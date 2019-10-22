<template>
    <div class="background">
        <span>홈페이지</span>
        <br>
        <v-flex v-for="stock in getStocks" :key="stock.id">
            <span>날짜: {{stock.date}}</span>
            <br>
            <span>시작가: {{stock.start_price}}</span>
            <br>
            <span>종가: {{stock.end_price}}</span>
            <br>
            <span>고가: {{stock.high_price}}</span>
            <br>
            <span>저가: {{stock.low_price}}</span>
            <br>
            <span>거래량: {{stock.val}}</span>
            <br>
            <button v-on:click="deleteStock(stock)">삭제</button>
        </v-flex>
    </div>
</template>

<script>
    import MainRepository from '../vuex/MainRepository'

    export default {
        name: "HomePage",
        components: {},
        data() {
            return {

            }
        },
        mounted() {
            MainRepository.setStocks();
        },
        methods: {
            deleteStock: function (id) {
                MainRepository.deleteStock(id);
            }
        },
        computed: {
            getStocks: function () {
                return MainRepository.getStocks();
            }
        },
    }
</script>

<style scoped>
    .background {
        height: 100vh;
        background-color: beige;
    }
</style>