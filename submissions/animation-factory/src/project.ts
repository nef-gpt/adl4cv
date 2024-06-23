import {makeProject} from '@motion-canvas/core';

import example from './scenes/example?scene';
import nn from './scenes/nn?scene';
import autoregressive from './scenes/autoregressive?scene';

import generation from "./scenes/generation?scene";

import "./global.css";

export default makeProject({
  scenes: [generation],
});
