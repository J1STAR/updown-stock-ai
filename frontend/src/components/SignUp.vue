<template>
    <div id="SignUp">
        <div> Sign Up</div>
        name : <input v-model="user.name" placeholder="name"> <br/>
        age : <input v-model="user.age" placeholder="ID"> <br/>
        <button v-on:click="postUser(user.name, user.age)">SignUp</button>
    </div>
</template>

<script>
    import MainRepository from '../vuex/MainRepository'

    export default {
        name: "SignUp",
        data() {
            return {
                user: {
                    name: '',
                    age: ' '
                }
            };
        },
        methods: {
            postUser: function (name, age) {
                MainRepository.postUser(name, age);
            }
        }

    }
</script>

<style scoped>

</style>