import {
	Node,
	NodeProps,
	Grid,
	Line,
	Rect,
	initial,
	signal,
} from "@motion-canvas/2d";
import {
	SignalValue,
	SimpleSignal,
	createSignal,
	all,
	tween,
	waitFor,
	createRef,
	Reference,
	makeRef,
} from "@motion-canvas/core";

export interface BarChartProps extends NodeProps {
	data: SignalValue<number[]>;
	range: SignalValue<{ min: number; max: number }>;
	height?: SignalValue<number>;
	barWidth?: SignalValue<number>;
	gap?: SignalValue<number>;
}

export class BarChart extends Node {
	@initial([])
	@signal()
	public declare readonly data: SimpleSignal<number[], this>;

	@initial({ min: 0, max: 1 })
	@signal()
	public declare readonly range: SimpleSignal<
		{ min: number; max: number },
		this
	>;

	@initial(300)
	@signal()
	public declare readonly height: SimpleSignal<number, this>;

	@initial(80)
	@signal()
	public declare readonly barWidth: SimpleSignal<number, this>;

	@initial(10)
	@signal()
	public declare readonly gap: SimpleSignal<number, this>;

	private bars: Reference<Rect>[] = [];

	// custom getter for the scaling of values

	private scalePosition(data: number, index: number) {
		const { min, max } = this.range();
		const barWidth = this.barWidth();
		const gap = this.gap();

		const indent = [10, 10] as const;
	}

	public constructor(props?: BarChartProps) {
		super({
			...props,
		});

		this.createGrid();
		this.createBars();
	}

	public createBars() {
		this.add(
			this.data().map((value, index, data) => {
				// console.log("create bar" + index + " " + value)
				return (
					<Rect
						ref={makeRef(this.bars, index)}
						fill={"#e6a700"}
						size={() => {
							const range = { min: Math.min(...data), max: Math.max(...data) };
							// console.log("calculate height" + this.height());
							return [
								this.barWidth(),
								((value - range.min) / (range.max - range.min)) * this.height(),
							] as [number, number];
						}}
						position={() => {
							const barWidth = this.barWidth();
							const gap = this.gap();
							const indent = [10, 10] as const;
							const range = { min: Math.min(...data), max: Math.max(...data) };
							const barHeight =
								((value - range.min) / (range.max - range.min)) * this.height();

							return [
								barWidth / 2 + index * (barWidth + gap) + indent[0],
								-barHeight / 2 - indent[1],
							] as [number, number];
						}}
					/>
				);
			})
		);
	}

	private createGrid() {
		// Add Grid background
		this.add(<Grid stroke={"#ccc"} lineWidth={1} size="800%" spacing={60} />);

		// Add Axes
		this.add(
			<Line
				points={() => [
					[0, 0],
					[this.data().length * (this.barWidth() + this.gap()), 0],
				]}
				stroke={"#fff"}
				lineWidth={4}
			/>
		);

		this.add(
			<Line
				points={[
					[0, -this.height()],
					[0, 0],
				]}
				stroke={"#fff"}
				lineWidth={4}
			/>
		);
	}
}
