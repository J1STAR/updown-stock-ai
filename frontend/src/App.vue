<template>
    <v-container>
        <SignUp/>
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
