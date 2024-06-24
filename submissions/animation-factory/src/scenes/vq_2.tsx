import { makeScene2D, Txt, Line } from '@motion-canvas/2d';
import { createRef, all, waitFor } from '@motion-canvas/core';
import { Rect } from '@motion-canvas/2d/lib/components';

const offset_x = -400;
const offset_y = -800;

const factor = 4

const tokenPositions = [0, factor*55, factor*110, factor*165, factor*220];
const tokenColors = ['#21918c', '#21918c', '#21918c', '#21918c', '#21918c', '#21918c'];
const scalarNames = ["0.1", "-0.1", "0.4", "0.3", "0.42", "-0.2"];
const tokenNames = ["ID.2", "ID.1", "ID.6", "ID.3", "ID.6", "ID.4"];
const tokenSpecialSOS = "SOS"
const tokenSpecialC = "C"
const specialColor = "#1a2145"

const novelColor = '#3b528b'
const otherNovelColor = "#3b528b"

const token_box_grid = factor*55;
const token_length = factor*50;

const xMovements = [0, 0]; // Add more if needed
const yMovements = [0, 6 * token_box_grid]; // Add more if needed

export default makeScene2D(function* (view) {
    // Create references for the boxes and lines
    const transformerBox = createRef<Rect>();
    const transformerText = createRef<Text>();
    const tokenRects = tokenPositions.map(() => createRef<Rect>());
    const tokenTexts = tokenPositions.map(() => createRef<Txt>());
    const tokenVerLines = tokenPositions.map(() => createRef<Line>());
    const novelToken = createRef<Rect>()
    const novelText = createRef<Txt>()
    const SOSToken = createRef<Rect>()
    const SOSText = createRef<Txt>()
    const CToken = createRef<Rect>()
    const CText = createRef<Txt>()

    // Create the initial input box and lines
    tokenVerLines.forEach((verLine, i) => {
        view.add(
            <Line
                ref={tokenVerLines[i]}
                points={[[tokenPositions[i] + offset_x, offset_y + yMovements[0] + token_length/2],
                [tokenPositions[i] + offset_x, offset_y + yMovements[yMovements.length - 1] - token_length/2]]}
                stroke="#FFFFFF"
                lineWidth={4}
                opacity={1}
            />
        );

    });

    // Create the initial input box and lines
    tokenPositions.forEach((x, i) => {

        view.add(
            <Rect
                ref={tokenRects[i]}
                width={token_length}
                opacity={1}
                height={token_length}
                fill={otherNovelColor}
                radius={10}
                position={[tokenPositions[i] + offset_x, offset_y + yMovements[yMovements.length - 1]]}
            />
        );
        view.add(
            <Txt
                text={tokenNames[i]}
                ref={tokenTexts[i]}
                fill="#ffffff"
                fontSize={factor*20}
                opacity={1}
                position={[tokenPositions[i] + offset_x, offset_y + yMovements[yMovements.length - 1]]}
                fontFamily={"Noto Sans Math"}
            />
        );

    });

    view.add(
        <Rect
            ref={novelToken}
            width={token_length}
            opacity={0}
            height={token_length}
            fill={novelColor}
            radius={10}
            position={[offset_x, offset_y]}
        />
    );

    view.add(
        <Txt
            ref={novelText}
            text={"ppeace"}
            fill="#FFFFFF"
            fontSize={factor*20}
            opacity={0}
            position={[300, 0]}
            fontFamily={"Noto Sans Math"}
        />
    );

    // Create the transformer box
    view.add(
        <Rect
            ref={transformerBox}
            width={(tokenPositions.length + 1) * token_box_grid}
            height={yMovements[yMovements.length - 1]/2 + token_box_grid/2}
            fill="#5ec962"
            radius={10}
            position={[2*token_box_grid + offset_x, 4*token_box_grid + offset_y - token_box_grid/1]}
        />
    );

    // Add text to the transformer box
    view.add(
        <Txt
            ref={transformerText}
            text="Vector Quantization"
            fill="#ffffff"
            fontSize={factor*20}
            position={[2*token_box_grid + offset_x, 4*token_box_grid + offset_y - token_box_grid/1]}
            fontFamily={"Inter"}
        />
    );


    // Create the transformer box
    view.add(
        <Rect
            ref={SOSToken}
            width={token_length}
            height={token_length}
            fill={specialColor}
            radius={10}
            position={[tokenPositions[0] + offset_x - 2*token_box_grid, yMovements[yMovements.length - 1] + offset_y]}
            opacity = {0} 
        />
    );

    // Add text to the transformer box
    view.add(
        <Txt
            ref={SOSText}
            text={tokenSpecialSOS}
            fill="#ffffff"
            fontSize={factor*20}
            position={[tokenPositions[0] + offset_x - 2*token_box_grid, yMovements[yMovements.length - 1] + offset_y]}
            fontFamily={"Inter"}
            opacity = {0} 
        />
    );

    // Create the transformer box
    view.add(
        <Rect
            ref={CToken}
            width={token_length}
            height={token_length}
            fill={specialColor}
            radius={10}
            position={[tokenPositions[0] + offset_x - token_box_grid, yMovements[yMovements.length - 1] + offset_y]}
            opacity = {0} 
        />
    );

    // Add text to the transformer box
    view.add(
        <Txt
            ref={CText}
            text={tokenSpecialC}
            fill="#ffffff"
            fontSize={factor*20}
            position={[tokenPositions[0] + offset_x - token_box_grid, yMovements[yMovements.length - 1] + offset_y]}
            fontFamily={"Inter"}
            opacity = {0} 
        />
    );

    yield* all(
        SOSToken().opacity(1, 1),
        SOSText().opacity(1, 1),
        CToken().opacity(1, 1),
        CText().opacity(1, 1), 
    );
    yield* waitFor(1.5);
});
