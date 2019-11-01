<template>
    <v-footer
            dark
            padless
    >
        <v-card
                flat
                tile
                class="indigo lighten-1 white--text text-center"
        >
            <v-container fluid>
                <v-row dense>
                    <v-col
                            v-for="card in cards"
                            :key="card.title"
                            xs6 sm6 md4
                    >
                        <v-card>
                            <v-img
                                    :src="card.src"
                                    class="white--text align-end"
                                    gradient="to bottom, rgba(0,0,0,.1), rgba(0,0,0,.5)"
                                    height="200px"
                            >
                                <v-card-title v-text="card.title"></v-card-title>
                                <v-card-text text-align-left>
                                    test
                                </v-card-text>
                            </v-img>

                            <v-card-actions>
                                <v-spacer></v-spacer>

                                <v-btn icon>
                                    <v-icon>mdi-github-box</v-icon>
                                </v-btn>
                            </v-card-actions>
                        </v-card>
                    </v-col>
                </v-row>
            </v-container>

            <v-divider></v-divider>

            <v-card-text class="white--text">
                {{ new Date().getFullYear() }} — <strong>Team.Up&Down</strong>
            </v-card-text>
        </v-card>
    </v-footer>
</template>

<script>
    export default {
        name: "TeamInfo",
        data: () => ({
            cards: [
                { title: '정우성', src: 'https://cdn.vuetifyjs.com/images/cards/house.jpg' },
                { title: '고민재', src: 'https://cdn.vuetifyjs.com/images/cards/house.jpg' },
                { title: '장한별', src: 'https://avatars3.githubusercontent.com/u/8870540?s=460&v=4' },
                { title: '정재욱', src: 'https://cdn.vuetifyjs.com/images/cards/house.jpg' },
                { title: '최병길', src: 'https://cdn.vuetifyjs.com/images/cards/house.jpg' },
                { title: '최석환', src: 'https://cdn.vuetifyjs.com/images/cards/house.jpg' },
            ],
        }),
    }
</script>

<style scoped>

</style>