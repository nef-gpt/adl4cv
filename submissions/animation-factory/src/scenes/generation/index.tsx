import { makeScene2D, Circle, Camera } from "@motion-canvas/2d";
import {
	all,
	createComputed,
	createComputedAsync,
	createRef,
	createSignal,
	tween,
	waitFor,
} from "@motion-canvas/core";
import { BarChart } from "./bar";
import { loadData } from "./data";

// Example usage
export default makeScene2D(function* (view) {
	const pathSignal = createSignal("media/logits_last.npy");
	const dataSignal = createComputedAsync(async () => {
		const data = await loadData(pathSignal());
		// @ts-ignore
		const result = Array.from(data.data);
		return result;
	}, []);

	const temperatureSignal = createSignal(1);
	const topKSignal = createSignal<number | null>(null);
	const transformedDataSignal = createComputed(() => {
		const data = dataSignal();
		const temperature = temperatureSignal();

		console.log("step ", temperature);

		// sort data
		const sortedData = data.slice().sort((a, b) => b - a);


		const topK = topKSignal();

		const tempData = sortedData.map((d) => Math.exp(d / temperature));

		// set all but top k to 0
		const topKData = topK === null ? tempData : tempData.slice(0, topK);
		const restData = topK === null ? [] : tempData.slice(topK);

		const transformedData = topKData.concat(restData);

		const normalization = transformedData.reduce((a, b) => a + b, 0);
		return transformedData.map((d) => d / normalization);
	});

	yield dataSignal;

	const rangeSignal = createComputed(() => {
		const data = transformedDataSignal();
		return { min: Math.min(...data), max: Math.max(...data) };
	});

	const camera = createRef<Camera>();
	const barChart = createRef<BarChart>();

	view.add(
		<Camera ref={camera}>
			<BarChart
				ref={barChart}
				data={transformedDataSignal}
				range={rangeSignal}
				height={400}
				barWidth={10}
				gap={3}
			/>
		</Camera>
	);

	yield* camera().centerOn([1650, -300], 0);
	yield* camera().zoom(0.28, 0);

	yield* temperatureSignal(400, 1);
});
