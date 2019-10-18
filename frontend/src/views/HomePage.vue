<template>
    <div class="background">
        <span>홈페이지</span>
        <br>
        <v-flex v-for="user in getUsers" :key="user">
            <span>이름: {{user.name}}</span>
            <br>
            <span>나이: {{user.age}}</span>
            <br>
        </v-flex>
    </div>
</template>

<script>
    import MainRepository from '../vuex/MainRepository'

    export default {
        name: "HomePage",
        components: {

        },
        data() {
            return {
                result: []
            }
        },
        mounted() {
            // this.getUsersMethod();
            MainRepository.setUsers();
        },
        methods: {
            getUsersMethod: function() {
                this.$http.get('/api').then((res) => {
                    console.log(res.data)
                })
            }
        },
        computed: {
            getUsers: function () {
                return MainRepository.getUsers();
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