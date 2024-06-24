import {makeProject} from '@motion-canvas/core';

import example from './scenes/example?scene';
import nn from './scenes/nn?scene';
import autoregressive from './scenes/autoregressive?scene';
import trial from './scenes/trial?scene';
import vq from './scenes/vq?scene';
import vq_2 from './scenes/vq_2?scene';


import "./global.css";

export default makeProject({
  scenes: [vq],
});
