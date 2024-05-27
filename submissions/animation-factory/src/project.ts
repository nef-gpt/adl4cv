import {makeProject} from '@motion-canvas/core';

import example from './scenes/example?scene';
import nn from './scenes/nn?scene';

export default makeProject({
  scenes: [nn],
});
