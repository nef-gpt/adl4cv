import { makeScene2D } from '@motion-canvas/2d';
import { Circle, Line } from '@motion-canvas/2d/lib/components';
import { createRef, Reference } from '@motion-canvas/core/lib/utils';
import { all, waitUntil } from '@motion-canvas/core/lib/flow';


export default makeScene2D(function* (view) {
  // Define the positions of the neurons
  const inputLayer = [-100, 100].map(x => ({ x, y: -300 }));
  const hiddenLayer = [-300, -100, 100, 300].map(x => ({ x, y: -100 }));
  const hiddenLayer2 = [-300, -100, 100, 300].map(x => ({ x, y: 100 }));
  const outputLayer = [0].map(x => ({ x, y: 300 }));

  // Create references for the neurons
  const inputNeurons = inputLayer.map(() => createRef<Circle>());
  const hiddenNeurons = hiddenLayer.map(() => createRef<Circle>());
  const hiddenNeurons2 = hiddenLayer2.map(() => createRef<Circle>());
  const outputNeuron = createRef<Circle>();

  // Add neurons to the scene
  inputLayer.forEach((pos, i) => {
    view.add(
      <Circle
        ref={inputNeurons[i]}
        x={pos.x}
        y={pos.y}
        width={50}
        height={50}
        fill="#3b528b"
        stroke="#fff"
        zIndex={5}
      />
    );
    // add a white circle slightly bigger to make the neuron look like it has a border
    view.add(
      <Circle
        x={pos.x}
        y={pos.y}
        width={60}
        height={60}
        fill="#fff"
        zIndex={4}
      />
    );
  });

  hiddenLayer.forEach((pos, i) => {
    view.add(
      <Circle
        ref={hiddenNeurons[i]}
        x={pos.x}
        y={pos.y}
        width={50}
        height={50}
        fill="#21918c"
        zIndex={5}
      />
    );
    // add a white circle slightly bigger to make the neuron look like it has a border
    view.add(
      <Circle
        x={pos.x}
        y={pos.y}
        width={60}
        height={60}
        fill="#fff"
        zIndex={4}
      />
    );

  });

  hiddenLayer2.forEach((pos, i) => {
    view.add(
      <Circle
      
        ref={hiddenNeurons2[i]}
        x={pos.x}
        y={pos.y}
        width={50}
        height={50}
        fill="#21918c"
        zIndex={5}
      />
    );
    // add a white circle slightly bigger to make the neuron look like it has a border
    view.add(
      <Circle
        x={pos.x}
        y={pos.y}
        width={60}
        height={60}
        fill="#fff"
        zIndex={4}
      />
    );
  });

  view.add(
    <Circle
      ref={outputNeuron}
      x={outputLayer[0].x}
      y={outputLayer[0].y}
      width={50}
      height={50}
      fill="#3b528b"
      zIndex={5}
    />
  );
  // add a white circle slightly bigger to make the neuron look like it has a border
  view.add(
    <Circle
      x={outputLayer[0].x}
      y={outputLayer[0].y}
      width={60}
      height={60}
      fill="#fff"
      zIndex={4}
    />
  );

  // Create edges between layers
  const edges = [];

  inputNeurons.forEach(input => {
    hiddenNeurons.forEach(hidden => {
      const edge = createRef<Line>();
      edges.push(edge);
      view.add(
        <Line
          ref={edge}
          points={[
            [input().x(), input().y()],
            [hidden().x(), hidden().y()],
          ]}
          stroke="#fff"
          lineWidth={2}
        />
      );
    });
  });

  hiddenNeurons.forEach(hidden => {
    hiddenNeurons2.forEach(hidden2 => {
      const edge = createRef<Line>();
      edges.push(edge);
      view.add(
        <Line
          ref={edge}
          points={[
            [hidden().x(), hidden().y()],
            [hidden2().x(), hidden2().y()],
          ]}
          stroke="#fff"
          lineWidth={2}
        />
      );
    });
  });

  hiddenNeurons2.forEach(hidden2 => {
    const edge = createRef<Line>();
    edges.push(edge);
    view.add(
      <Line
        ref={edge}
        points={[
          [hidden2().x(), hidden2().y()],
          [outputNeuron().x(), outputNeuron().y()],
        ]}
        stroke="#fff"
        lineWidth={2}
      />
    );
  });

  // Add edges to the scene
  edges.forEach(edge => view.add(edge));

  const createNodeShuffleAnimation = (first: number, second: number) => {
    const [node1, node2] = [hiddenNeurons[first], hiddenNeurons[second]];
    const tempX = node1().x();
    const tempY = node1().y();

const time = 5;

return all(
      node1().position.x(node2().x(), 1),
      node1().position.y(node2().y(), 1),
      node2().position.x(tempX, 1),
      node2().position.y(tempY, 1),
      // Update edges connected to node1
      ...edges.filter(edge => edge().points()[1][0] === tempX && edge().points()[1][1] === tempY).map(edge =>
        edge().points([
          [edge().points()[0][0], edge().points()[0][1]],
          [node2().x(), node2().y()]
        ], time)
      ),
      // Update edges connected to node2
      ...edges.filter(edge => edge().points()[1][0] === node2().x() && edge().points()[1][1] === node2().y()).map(edge =>
        edge().points([
          [edge().points()[0][0], edge().points()[0][1]],
          [tempX, tempY]
        ], time)
      ),
      // Update edges connected *after* node1
      ...edges.filter(edge => edge().points()[0][0] === tempX && edge().points()[0][1] === tempY).map(edge =>
        edge().points([
          [node2().x(), node2().y()],
          [edge().points()[1][0], edge().points()[1][1]]
        ], time)
      ),
      // Update edges connected *after* node2
      ...edges.filter(edge => edge().points()[0][0] === node2().x() && edge().points()[0][1] === node2().y()).map(edge =>
        edge().points([
          [tempX, tempY],
          [edge().points()[1][0], edge().points()[1][1]]
        ], time)
      )
    );
  }

  // Shuffle nodes
  yield* createNodeShuffleAnimation(0, 1);
  yield* createNodeShuffleAnimation(2, 3);
  // yield* createNodeShuffleAnimation(0, 2);
  
  
});