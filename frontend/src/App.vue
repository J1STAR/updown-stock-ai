<template>
    <v-container>
        <SignUp/>
        <br>
        <router-view/>
    </v-container>
</template>

<script>
    // import HelloWorld from './components/HelloWorld';
    import SignUp from "./components/SignUp";

    export default {
        name: 'App',
        components: {
            SignUp
        },
        data: () => ({
            //
        }),
    };
</script>
