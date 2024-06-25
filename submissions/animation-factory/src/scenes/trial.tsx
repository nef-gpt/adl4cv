import { makeScene2D, Txt, Line } from '@motion-canvas/2d';
import { createRef, all, waitFor } from '@motion-canvas/core';
import { Rect } from '@motion-canvas/2d/lib/components';

const offset_x = -200;
const offset_y = -700;

const factor = 4

const tokenPositions = [0, factor*55, factor*110, factor*165, factor*220];
const tokenColors = ['#21918c', '#21918c', '#21918c', '#21918c', '#21918c', '#21918c'];
const tokenNames = ["SOS", "C", "ID.2", "ID.1", "ID.6", "ID.3"];
const novelColor = '#1a2145'
const otherNovelColor = "#21918c"//#3b528b"

const token_box_grid = factor*55;
const token_length = factor*50;

const xMovements = [0, token_box_grid, token_box_grid]; // Add more if needed
const yMovements = [token_box_grid, token_box_grid, 6 * token_box_grid]; // Add more if needed

export default makeScene2D(function* (view) {
    // Create references for the boxes and lines
    const transformerBox = createRef<Rect>();
    const transformerText = createRef<Text>();
    const tokenRects = tokenPositions.map(() => createRef<Rect>());
    const tokenTexts = tokenPositions.map(() => createRef<Txt>());
    const tokenVerLines1 = tokenPositions.map(() => createRef<Line>());
    const tokenVerLines2 = tokenPositions.map(() => createRef<Line>());
    const tokenHorLines = tokenPositions.map(() => createRef<Line>());
    const novelToken = createRef<Rect>()
    const novelText = createRef<Txt>()

    // Create the initial input box and lines
    tokenHorLines.forEach((horLine, i) => {
        view.add(
            <Line
                ref={tokenHorLines[i]}
                points={[[tokenPositions[i] + offset_x + xMovements[0], offset_y + yMovements[0]],
                [tokenPositions[i] + offset_x + xMovements[1], offset_y + yMovements[0]]]}
                stroke="#FFFFFF"
                lineWidth={4}
                opacity={0}
            />
        );

        view.add(
            <Line
                ref={tokenVerLines1[i]}
                points={[[tokenPositions[i] + offset_x, offset_y + token_length / 2],
                [tokenPositions[i] + offset_x, offset_y + yMovements[0]]]}
                stroke="#FFFFFF"
                lineWidth={4}
                opacity={0}
            />
        );
        view.add(
            <Line
                ref={tokenVerLines2[i]}
                points={[[tokenPositions[i] + offset_x + xMovements[1], offset_y + yMovements[0]],
                [tokenPositions[i] + offset_x + xMovements[1], offset_y + yMovements[yMovements.length - 1] - token_length / 2]]}
                stroke="#FFFFFF"
                lineWidth={4}
                opacity={0}
            />
        );


    });






    // Create the initial input box and lines
    tokenPositions.forEach((x, i) => {

        view.add(
            <Rect
                ref={tokenRects[i]}
                width={token_length}
                opacity={0}
                height={token_length}
                fill={tokenColors[i]}
                radius={10}
                position={[x + offset_x, offset_y]}
            />
        );
        view.add(
            <Txt
                text={tokenNames[i]}
                ref={tokenTexts[i]}
                fill="#ffffff"
                fontSize={factor*20}
                opacity={0}
                position={[x + offset_x, offset_y]}
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
            position={[3*token_box_grid + offset_x, 4*token_box_grid + offset_y - token_box_grid/1.5]}
        />
    );

    // Add text to the transformer box
    view.add(
        <Txt
            ref={transformerText}
            text="Transformer"
            fill="#ffffff"
            fontSize={factor*20}
            position={[3*token_box_grid + offset_x, 4*token_box_grid + offset_y - token_box_grid/1.5]}
            fontFamily={"Inter"}
        />
    );

    yield* all(
        ...tokenHorLines.map((horLine, i) =>
            horLine().opacity(0.2, 0.5),
        ),
        ...tokenVerLines1.map((verLine, i) =>
            verLine().opacity(0.2, 0.5),
        ),
        ...tokenVerLines2.map((verLine, i) =>
            verLine().opacity(0.2, 0.5),
        ),
    );



    for (let i = 0; i < tokenPositions.length; i++) {

        yield* all(
            tokenTexts[0]().position([tokenPositions[0] + offset_x, offset_y - 2*token_box_grid], 0),
            tokenRects[0]().position([tokenPositions[0] + offset_x, offset_y - 2*token_box_grid], 0),
        );

        yield* all(
            tokenTexts[0]().position([tokenPositions[0] + offset_x, offset_y], 1),
            tokenRects[0]().position([tokenPositions[0] + offset_x, offset_y], 1),
            tokenHorLines[i]().opacity(1, 0.5),
            tokenVerLines1[i]().opacity(1, 0.5),
            tokenVerLines2[i]().opacity(1, 0.5),
            ...tokenTexts.map((tokenText, k) => {
                if (k < i + 1) {
                    return tokenText().opacity(1, 0.5);
                }
            }).filter(Boolean),
            ...tokenRects.map((tokenRect, k) => {
                if (k < i + 1) {
                    return tokenRect().opacity(1, 0.5);
                }
            }).filter(Boolean),
        );

        // do all movement for all current tokens
        for (let j = 0; j < xMovements.length; j++) {
            yield* all(
                ...tokenRects.map((tokenRect, k) => {
                    if (k < i + 1) {
                        return tokenRect().position([tokenPositions[k] + xMovements[j] + offset_x, offset_y + yMovements[j]], 1);
                    }
                }).filter(Boolean),
                ...tokenTexts.map((tokenText, k) => {
                    if (k < i + 1) {
                        return tokenText().position([tokenPositions[k] + xMovements[j] + offset_x, offset_y + yMovements[j]], 1);
                    }
                }).filter(Boolean),
                ...tokenTexts.map((tokenText, k) => {
                    if (k < i + 1) {
                        if (j == xMovements.length - 1) {
                            return tokenText().opacity(0, 0.5);
                        }
                    }
                }).filter(Boolean),
                ...tokenRects.map((tokenRect, k) => {
                    if (k < i + 1) {
                        if (j == xMovements.length - 1) {
                            return tokenRect().opacity(0, 0.5);
                        }
                    }
                }).filter(Boolean),

            );
            yield* waitFor(0.5);
        }

        const shakeVal = 2;
        for (let i = 0; i < 3; i++) {
            yield* all(
                transformerBox().rotation(shakeVal, 0.075),
        )
            yield* transformerBox().rotation(-shakeVal, 0.06);
        }
        yield* transformerBox().rotation(0, 0.05);

        for (let j = 0; j < tokenRects.length; j++) {
            yield* tokenRects[j]().fill(otherNovelColor, 0.0);
        } 

        yield* novelToken().position([tokenPositions[i] + xMovements[xMovements.length - 1] + offset_x, offset_y + yMovements[yMovements.length - 1] - 2 * token_box_grid], 0);
        yield* novelText().position([tokenPositions[i] + xMovements[xMovements.length - 1] + offset_x, offset_y + yMovements[yMovements.length - 1] - 2 * token_box_grid], 0);
        yield* novelText().text(tokenNames[i + 1]);

        for (let j = 0; j < i; j++) {
            yield* tokenTexts[j]().text(tokenNames[j + 1]);
        }


        yield* all(
            ...tokenRects.map((tokenRect, k) => {
                if (k < i + 1) {
                    return tokenRect().position([tokenPositions[k] + xMovements[xMovements.length - 1] + offset_x, offset_y + yMovements[yMovements.length - 1] - 2 * token_box_grid], 0);
                }
            }).filter(Boolean),
            ...tokenTexts.map((tokenText, k) => {
                if (k < i + 1) {
                    return tokenText().position([tokenPositions[k] + xMovements[xMovements.length - 1] + offset_x, offset_y + yMovements[yMovements.length - 1] - 2 * token_box_grid], 0);
                }
            }).filter(Boolean),

        );

        yield* all(
            ...tokenRects.map((tokenRect, k) => {
                if (k < i + 1) {
                    return tokenRect().position([tokenPositions[k] + xMovements[xMovements.length - 1] + offset_x, offset_y + yMovements[yMovements.length - 1]], 1);
                }
            }).filter(Boolean),
            ...tokenTexts.map((tokenText, k) => {
                if (k < i + 1) {
                    return tokenText().position([tokenPositions[k] + xMovements[xMovements.length - 1] + offset_x, offset_y + yMovements[yMovements.length - 1]], 1);
                }
            }).filter(Boolean),
            novelToken().position([tokenPositions[i] + xMovements[xMovements.length - 1] + offset_x, offset_y + yMovements[yMovements.length - 1]], 1),
            novelText().position([tokenPositions[i] + xMovements[xMovements.length - 1] + offset_x, offset_y + yMovements[yMovements.length - 1]], 1),
            novelText().opacity(1, 0),
            novelToken().opacity(1, 0),

            ...tokenTexts.map((tokenText, k) => {
                if (k < i) {
                    return tokenText().opacity(1, 0.5);
                }
            }).filter(Boolean),
            ...tokenRects.map((tokenRect, k) => {
                if (k < i) {
                    return tokenRect().opacity(1, 0.5);
                }
            }).filter(Boolean),
        );

        if (i == tokenPositions.length - 1) {
            continue;
        }


        yield* all(
            ...tokenRects.map((tokenRect, k) => {
                if (k < i + 1) {
                    return tokenRect().position([tokenPositions[k] + xMovements[xMovements.length - 1] + offset_x - (xMovements[1] + (i + 1) * token_box_grid), offset_y + yMovements[yMovements.length - 1]], 1);
                }
            }).filter(Boolean),
            ...tokenTexts.map((tokenText, k) => {
                if (k < i + 1) {
                    return tokenText().position([tokenPositions[k] + xMovements[xMovements.length - 1] + offset_x - (xMovements[1] + (i + 1) * token_box_grid), offset_y + yMovements[yMovements.length - 1]], 1);
                }
            }).filter(Boolean),
            novelToken().position([tokenPositions[i] + xMovements[xMovements.length - 1] + offset_x - (xMovements[1] + (i + 1) * token_box_grid), offset_y + yMovements[yMovements.length - 1]], 1),
            novelText().position([tokenPositions[i] + xMovements[xMovements.length - 1] + offset_x - (xMovements[1] + (i + 1) * token_box_grid), offset_y + yMovements[yMovements.length - 1]], 1),

        );

        yield* all(
            ...tokenRects.map((tokenRect, k) => {
                if (k < i + 1) {
                    return tokenRect().position([tokenPositions[k] + xMovements[xMovements.length - 1] + offset_x - (xMovements[1] + (i + 1) * token_box_grid), offset_y], 1);
                }
            }).filter(Boolean),
            ...tokenTexts.map((tokenText, k) => {
                if (k < i + 1) {
                    return tokenText().position([tokenPositions[k] + xMovements[xMovements.length - 1] + offset_x - (xMovements[1] + (i + 1) * token_box_grid), offset_y], 1);
                }
            }).filter(Boolean),
            novelToken().position([tokenPositions[i] + xMovements[xMovements.length - 1] + offset_x - (xMovements[1] + (i + 1) * token_box_grid), offset_y], 1),
            novelText().position([tokenPositions[i] + xMovements[xMovements.length - 1] + offset_x - (xMovements[1] + (i + 1) * token_box_grid), offset_y], 1),

        );

        yield* all(
            ...tokenRects.map((tokenRect, k) => tokenRect().position([tokenPositions[k + 1] + offset_x, offset_y], 1)),
            ...tokenTexts.map((tokenText, k) => tokenText().position([tokenPositions[k + 1] + offset_x, offset_y], 1)),
            novelToken().position([tokenPositions[i + 1] + offset_x, offset_y], 1),
            novelText().position([tokenPositions[i + 1] + offset_x, offset_y], 1),

        );



        yield* all(
            ...tokenRects.map((tokenRect, k) => tokenRect().opacity(0, 0.5)),
            ...tokenTexts.map((tokenText, k) => tokenText().opacity(0, 0.5)),
            novelToken().opacity(0, 0.5),
            novelText().opacity(0, 0.5),
        );

        yield* all(
            ...tokenRects.map((tokenRect, k) => tokenRect().position([tokenPositions[k] + offset_x, offset_y], 0)),
            ...tokenTexts.map((tokenText, k) => tokenText().position([tokenPositions[k] + offset_x, offset_y], 0)),
        );

        for (let i = 0; i < tokenRects.length; i++) {
            tokenTexts[i]().text(tokenNames[i]);
        }

        for (let j = 0; j < tokenRects.length; j++) {
            yield* tokenRects[j]().fill(tokenColors[j], 0.0);
        } 

        yield* waitFor(0.5);
    }
    yield* waitFor(1.5);
});
