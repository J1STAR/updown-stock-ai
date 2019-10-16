<template>
	<v-app>
		<v-content>
			<v-container fluid fill-height>
				<v-layout>
					<v-flex
							xs2
					>

					</v-flex>
					<v-flex
							xs8
					>
						<LineChart/>
					</v-flex>
					<v-flex
							xs2
					>

					</v-flex>
				</v-layout>
			</v-container>
		</v-content>
	</v-app>
</template>

<script>
	import LineChart from "@/components/LineChart";

	export default {
		name: 'App',
		components: {
			LineChart
		},
		data: () => ({
			//
		}),
	};
</script>
