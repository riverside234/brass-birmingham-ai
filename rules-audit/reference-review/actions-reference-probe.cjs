// Runs the pinned independent reference locally. Records a limitation in that
// reference, not a defect in the current Steam DLL.
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const root = path.resolve(__dirname, '../..');
const reference = path.join(root, 'artifacts/reference-npow-brass/js');
const context = vm.createContext({console});
for (const file of ['gameData.js', 'gameState.js', 'gameLogic.js']) {
  vm.runInContext(fs.readFileSync(path.join(reference, file), 'utf8'), context, {filename:file});
}
const result = vm.runInContext(`(() => {
  const state = new GameState(2, ['A', 'B']);
  state.era = ERA.RAIL;
  state.boardLinks['birmingham-redditch'] = {playerId:0,type:'rail'};
  const mine = () => ({playerId:0,type:INDUSTRY_TYPES.COAL_MINE,flipped:false,
    resourceCubes:2,tileData:{level:2,income:7}});
  state.boardIndustries['dudley_0'] = mine();
  state.boardIndustries['redditch_0'] = mine();
  const logic = new GameLogic(state);
  const link = CONNECTIONS.find(c => c.id === 'birmingham-dudley');
  const selectedBeforePlacement = logic.findCheapestCoalForLink(link,0);
  state.boardLinks[link.id] = {playerId:0,type:'rail'};
  const selectedAfterPlacement = logic.findCheapestCoalForLink(link,0);
  delete state.boardLinks[link.id];
  const execution = logic.executeNetwork(0,link.id,0);
  return {expectedNearest:'dudley_0',selectedBeforePlacement,selectedAfterPlacement,
    execution,remainingDudley:state.boardIndustries['dudley_0'].resourceCubes,
    remainingRedditch:state.boardIndustries['redditch_0'].resourceCubes,
    referenceHasNearestEndpointDefect:selectedBeforePlacement.key !== 'dudley_0'};
})()`, context);
const report = {revision:'fdeb63c6aa92cc4ba44a721482316e0b8db10e06',...result};
fs.writeFileSync(path.join(__dirname,'actions-reference-results.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
