<template>
    <div id="SignUp">
        <div> Sign Up</div>
        name : <input v-model="user.name" placeholder="name"> <br/>
        age : <input v-model="user.age" placeholder="ID"> <br/>
        <button v-on:click="createUser">SignUp</button>
    </div>
</template>

<script>
    export default {
        name: "SignUp",
        data() {
            return {
                user: {
                    name: '',
                    age: 0
                }
            };
        },
        methods: {
            createUser: function() {
              this.$http.post('/api/', {name: this.user.name, age: this.user.age})
            }
        }

    }
</script>

<style scoped>

</style>
