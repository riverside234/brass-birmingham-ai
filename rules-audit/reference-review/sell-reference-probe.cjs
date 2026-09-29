// Independent-reference limitation. Does not load or change the Steam game.
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const reference = path.resolve(__dirname, '../../artifacts/reference-npow-brass/js');
const context = vm.createContext({});
for (const name of ['gameData.js', 'gameState.js', 'gameLogic.js']) {
  vm.runInContext(fs.readFileSync(path.join(reference, name), 'utf8'), context, {filename:name,timeout:1000});
}
const result = vm.runInContext(`(() => {
  const state = new GameState(2, ['A','B']);
  const logic = new GameLogic(state);
  state.boardLinks['birmingham-oxford'] = {playerId:0,type:'canal'};
  state.merchantTiles = [0,1].map(() => ({location:'oxford',buys:INDUSTRY_TYPES.MANUFACTURER,hasBeer:true}));
  const data = INDUSTRY_DATA[INDUSTRY_TYPES.MANUFACTURER].find(t => t.level === 5);
  state.boardIndustries.birmingham_0 = {playerId:0,type:INDUSTRY_TYPES.MANUFACTURER,tileData:data,flipped:false,resourceCubes:0};
  const before = {beer:state.merchantTiles.filter(t => t.hasBeer).length,income:state.players[0].incomeSpace,hand:state.players[0].hand.length};
  const offered = logic.getValidSellTargets(0).some(t => t.key === 'birmingham_0');
  const execution = logic.executeSell(0,[{key:'birmingham_0',useMerchantBeer:true}],0);
  return {offered,execution,before,after:{beer:state.merchantTiles.filter(t => t.hasBeer).length,income:state.players[0].incomeSpace,hand:state.players[0].hand.length,flipped:state.boardIndustries.birmingham_0.flipped}};
})()`, context, {timeout:1000});
const report = {revision:'fdeb63c6aa92cc4ba44a721482316e0b8db10e06',scope:'reference only',...result};
fs.writeFileSync(path.join(__dirname,'sell-reference-results.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
