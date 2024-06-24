import {
  Node,
  NodeProps,
  Grid,
  Line,
  Rect,
  initial,
  signal,
} from '@motion-canvas/2d';
import {
  SignalValue,
  SimpleSignal,
  createSignal,
  all,
  tween,
  waitFor,
  createRef,
} from '@motion-canvas/core';

export interface BarChartProps extends NodeProps {
  data: SignalValue<number[]>;
  range: SignalValue<{ min: number; max: number }>;
}

export class BarChart extends Node {
  @initial([])
  @signal()
  public declare readonly data: SimpleSignal<number[], this>;

  @initial({ min: 0, max: 1 })
  @signal()
  public declare readonly range: SimpleSignal<{ min: number; max: number }, this>;

  private bars: Rect[] = [];

  public constructor(props?: BarChartProps) {
    super({
      ...props,
    });
  }

  public *updateData(newData: number[], duration: number) {
    const oldData = this.data();
    this.data(newData);

    const { min, max } = this.range();
    const barWidth = 50;
    const gap = 10;
    const scale = (value: number) => ((value - min) / (max - min)) * 100;

    const animations = [];

    // Update or add bars
    newData.forEach((value, index) => {
      if (index < this.bars.length) {
        // Update existing bar
        const bar = this.bars[index];
        animations.push(
          tween(duration, t => {
            bar.size([barWidth, scale(value) * t + bar.size().y * (1 - t)]);
            bar.position([
              index * (barWidth + gap) - (newData.length * (barWidth + gap)) / 2,
              -scale(value) / 2,
            ]);
          })
        );
      } else {
        // Add new bar
        const bar = createRef<Rect>();
        this.add(
          <Rect
            ref={bar}
            fill={"#e6a700"}
            size={[barWidth, 0]} // Start with height 0
            position={[
              index * (barWidth + gap) - (newData.length * (barWidth + gap)) / 2,
              0,
            ]}
          />
        );
        this.bars.push(bar());
        // Animate the height from 0 to the scaled value
        animations.push(
          tween(duration, t => {
            bar().size([barWidth, scale(value) * t]);
            bar().position([
              index * (barWidth + gap) - (newData.length * (barWidth + gap)) / 2,
              -scale(value) / 2,
            ]);
          })
        );
      }
    });

    // Remove extra bars
    while (this.bars.length > newData.length) {
      const bar = this.bars.pop();
      if (bar) {
        // Animate the fade-out and height reduction
        animations.push(
          tween(duration, t => {
            bar.size([barWidth, bar.size().y * (1 - t)]);
            bar.opacity(1 - t);
          })
        );
      }
    }

    // Run all animations in parallel
    yield* all(...animations);

    // remove extra bars
    while (this.bars.length > newData.length) {
      const bar = this.bars.pop();
      if (bar) {
        this.removeChild(bar);
      }
    }

  }
}

