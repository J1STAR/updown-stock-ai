<template>
    <div id="SignUp">
        <div> Sign Up</div>
        날짜: <input v-model="stock.date" placeholder="date"> <br/>
        시작가 : <input v-model="stock.start_price" placeholder="start_price"> <br/>
        고가: <input v-model="stock.high_price" placeholder="high_price"> <br/>
        저가: <input v-model="stock.low_price" placeholder="low_price"> <br/>
        종가: <input v-model="stock.end_price" placeholder="end_price"> <br/>
        val: <input v-model="stock.val" placeholder="val"> <br/>
        <button v-on:click="postStock(stock)">SignUp</button>
    </div>
</template>

<script>
    import MainRepository from '../vuex/MainRepository'

    export default {
        name: "SignUp",
        data() {
            return {
                stock: {
                    date: '',
                    start_price: ' ',
                    end_price: ' ',
                    high_price: ' ',
                    low_price: ' ',
                    val: ' '
                }
            };
        },
        methods: {
            postStock: function (stock) {
                MainRepository.postStock(stock);
            }
        }

    }
</script>

<style scoped>

</style>