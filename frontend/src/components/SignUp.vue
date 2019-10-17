<template>
    <div id="SignUp">
        <div> Sign Up</div>
        name : <input v-model="user.name" placeholder="name"> <br/>
        age : <input v-model="user.age" placeholder="ID"> <br/>
        <button v-on:click="signUp">SignUp</button>
    </div>
</template>

<script>
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
            signUp() {
                // this.$http.post('/api/login/signUp', {
                //     user: this.user
                // }).then((response) => {
                //     if (response.data.result == 0) {
                //         alert('Error, try again');
                //     } else if (response.data.result == 1) {
                //         alert('Success');
                //         this.$router.push('login');
                //     }
                // }).catch(function (error) {
                //     alert(error);
                // })
            }
        }

    }
</script>

<style scoped>

</style>