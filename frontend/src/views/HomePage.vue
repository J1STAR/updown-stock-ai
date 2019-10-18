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
<<<<<<< HEAD
            // MainRepository.setUsers();
            this.loadUser();
        },
        methods: {
            loadUser: function() {
              this.$http.get('/api').then((res) => {
                console.log(res)
              })
=======
            // this.getUsersMethod();
            MainRepository.setUsers();
        },
        methods: {
            getUsersMethod: function() {
                this.$http.get('/api').then((res) => {
                    console.log(res.data)
                })
>>>>>>> b17d8142c95937e94a2d3babc9ae1e181213a81a
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
