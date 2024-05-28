import { makeScene2D, Txt } from '@motion-canvas/2d';
import { createRef, all, waitFor } from '@motion-canvas/core';
import { Rect, Circle, Node } from '@motion-canvas/2d/lib/components';

const tokens = [
  {
    x: -55,
    fill: '#21918c',
  },
  {
    x: 0,
    fill: '#21918c',
  },
  {
    x: 55,
    fill: '#21918c',
  },
]

const newTokenColor = "#3b528b"

export default makeScene2D(function* (view) {
  // Create references for the boxes
  const transformerBox = createRef<Rect>();
  const tokenRects = tokens.map(() => createRef<Rect>());
  const tokenTexts = tokens.map(() => createRef<Txt>());
  const token_names = ["θ₀", "θ₁", "θ₂"];


  // Create the initial input box
  tokens.forEach((token, i) => {
    view.add(
      <Rect
        ref={tokenRects[i]}
        width={50}
        opacity={0}
        height={50}
        fill={token.fill}
        radius={10}
        position={[token.x - 300, 0]}
      />
    );
    view.add(
      <Txt
        text={token_names[i]}
        ref={tokenTexts[i]}
        fill="#ffffff"
        fontSize={20}
        opacity={0}
        position={[token.x - 300, 0]}
        fontFamily={"Noto Sans Math"}
      />
    );
  });
  

  // Create the transformer box
  view.add(
    <Rect
      ref={transformerBox}
      width={250}
      height={100}
      fill="#5ec962"
      radius={10}
      position={[0, 0]}
    />
  );

  // Add text to the transformer box
  view.add(
    <Txt
      text="Transformer(θᵢ₋₁, θᵢ₋₂...θ₀)"
      fill="#ffffff"
      fontSize={20}
      position={[0, 0]}
      fontFamily={"Inter"}
    />
  );

  

  // Animation sequence

  // Show the first token with the label
  yield* all(
    tokenRects[0]().opacity(1, 0.5),
    tokenTexts[0]().opacity(1, 0.5)
  )



  for (let i = 0; i < 2; i++) {
    // Move input box to transformer
    yield* all(...tokenRects.map((tokenRect, i) => tokenRect().position([tokens[i].x, 0], 1)),
      ...tokenTexts.map((tokenText, i) => tokenText().position([tokens[i].x, 0], 1)));

    // Simulate processing in transformer
    // shake the transformer (eg. increate the rotation a bit and then decrease it back to normal for a few times)
    const shakeVal = 2;
    for (let i = 0; i < 3; i++) {
      yield* transformerBox().rotation(shakeVal, 0.05);
      yield* transformerBox().rotation(-shakeVal, 0.05);
    }
    yield* transformerBox().rotation(0, 0.05);

    // imidiately show the next token
    yield* tokenRects[i + 1]().opacity(1, 0);
    yield* tokenTexts[i + 1]().opacity(1, 0);
    // and change color of the new token
    yield* tokenRects[i + 1]().fill(newTokenColor, 0.5);

    // Move all tokens to the right
    yield* all(
      ...tokenRects.map((tokenRect, i) => tokenRect().position([tokens[i].x + 300, 0], 1)),
      ...tokenTexts.map((tokenText, i) => tokenText().position([tokens[i].x + 300, 0], 1))
    );

    // Wait for a bit
    yield* waitFor(0.5);

    // if it's the last iteration, don't move the tokens back
    if (i == 1) {
      continue;
    }

    // Move all tokens back to the left and home
    // First to 300, -100
    // Then to -300, -100
    // Then to -300, 0
    yield* all(
      ...tokenRects.map((tokenRect, i) => tokenRect().position([tokens[i].x + 300, -100], 0.5)),
      ...tokenTexts.map((tokenText, i) => tokenText().position([tokens[i].x + 300, -100], 0.5))
    );
    yield* all(...tokenRects.map((tokenRect, i) => tokenRect().position([tokens[i].x - 300, -100], 0.8)),
      ...tokenTexts.map((tokenText, i) => tokenText().position([tokens[i].x - 300, -100], 0.8)));
    yield* all(...tokenRects.map((tokenRect, i) => tokenRect().position([tokens[i].x - 300, 0], 0.5)),
      ...tokenTexts.map((tokenText, i) => tokenText().position([tokens[i].x - 300, 0], 0.5)));


    // now revert the color of the new token
    yield* tokenRects[i + 1]().fill(tokens[i + 1].fill, 0.5);

    // // Show output boxes
    // yield* all(
    //   outputBox1().opacity(1, 0.5),
    //   outputBox2().opacity(1, 0.5)
    // );

    // // Move output boxes to the right
    // yield* all(
    //   outputBox1().position([200 + i * 60, -30], 1),
    //   outputBox2().position([200 + i * 60, 30], 1)
    // );

    // Prepare for next iteration
    yield* waitFor(0.5);
  }

  // at the end move all tokens to the bottom of the screen
  yield* all(
    ...tokenRects.map((tokenRect, i) => tokenRect().position([tokens[i].x + 300, 400], 0.5)),
    ...tokenTexts.map((tokenText, i) => tokenText().position([tokens[i].x + 300, 400], 0.5))
  );
});