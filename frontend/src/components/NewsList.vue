<template>
    <v-container id="news-list-container">
        <v-layout>
            <v-flex>
                <v-card>
                    <v-list>
                        <v-subheader id="news_corp_name"><span>{{ news_target }}&nbsp;</span>관련 뉴스</v-subheader>

                        <template v-if="status">
                            <template v-if="news_list.length !== 0">
                                <v-list-item  v-for="news in news_list" :key="news.link">
                                    <v-list-item-content>
                                        <v-list-item-title @click="goToNewsLink(news.link)">
                                            {{ news.title }}
                                        </v-list-item-title>
                                    </v-list-item-content>
                                </v-list-item>
                            </template>
                            <template>
                                <v-list-item>
                                    <v-list-item-content>
                                        <v-list-item-title>
                                            Not Exist {{ news_target }} News List
                                        </v-list-item-title>
                                    </v-list-item-content>
                                </v-list-item>
                            </template>
                        </template>
                        <template v-else>
                            <v-list-item>
                                <v-list-item-content>
                                    <v-list-item-title align="center">
                                        <v-progress-circular
                                                :size="128"
                                                :width="10"
                                                color="lightskyblue"
                                                indeterminate
                                        ></v-progress-circular>
                                    </v-list-item-title>
                                </v-list-item-content>
                            </v-list-item>
                        </template>
                    </v-list>
                </v-card>
            </v-flex>
        </v-layout>
    </v-container>
</template>

<script>
    export default {
        name: "NewsList",
        props: {
            news_target: {
                type: String
            },
        },
        data() {
            return {
                news_list: [],
                status: false
            }
        },
        watch: {
            news_target: function(target) {
                this.loadNewsData(target)
            }
        },
        methods: {
            goToNewsLink: function(link) {
                let tab = window.open(link, '_blank');
                tab.focus();
            },
            loadNewsData: async function(target) {
                this.status = false
                this.news_list = await this.$http.get("/news/"+ target).then((res) => {
                    this.status = true
                    return res.data.news
                })
            }
        }
    }
</script>

<style scoped>
    #news_corp_name {
        color: lightskyblue;

        font-weight: bold;
        font-size: 16px
    }
</style>